from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional
import math

app = FastAPI(title="CineStar Booking API")


@app.get("/")
def welcome():
    return {"message": "Welcome to CineStar Booking"}

movies = [
    {"id": 1, "title": "Inception",                   "genre": "Action",  "language": "English", "duration_mins": 148, "ticket_price": 300, "seats_available": 120},
    {"id": 2, "title": "The Dark Knight",              "genre": "Action",  "language": "English", "duration_mins": 152, "ticket_price": 350, "seats_available": 80},
    {"id": 3, "title": "3 Idiots",                    "genre": "Comedy",  "language": "Hindi",   "duration_mins": 170, "ticket_price": 200, "seats_available": 150},
    {"id": 4, "title": "Tumbbad",                     "genre": "Horror",  "language": "Hindi",   "duration_mins": 104, "ticket_price": 250, "seats_available": 60},
    {"id": 5, "title": "The Shawshank Redemption",    "genre": "Drama",   "language": "English", "duration_mins": 142, "ticket_price": 280, "seats_available": 100},
    {"id": 6, "title": "Taare Zameen Par",             "genre": "Drama",   "language": "Hindi",   "duration_mins": 162, "ticket_price": 220, "seats_available": 90},
    {"id": 7, "title": "Get Out",                     "genre": "Horror",  "language": "English", "duration_mins": 104, "ticket_price": 260, "seats_available": 70},
    {"id": 8, "title": "Andhadhun",                   "genre": "Drama",   "language": "Hindi",   "duration_mins": 139, "ticket_price": 240, "seats_available": 110},
]

movie_id_counter = 9


@app.get("/movies")
def get_all_movies():
    total_seats = sum(m["seats_available"] for m in movies)
    return {
        "total": len(movies),
        "total_seats_available": total_seats,
        "movies": movies,
    }

bookings = []
booking_counter = 1


@app.get("/bookings")
def get_all_bookings():
    total_revenue = sum(b["total_cost"] for b in bookings)
    return {
        "total": len(bookings),
        "total_revenue": total_revenue,
        "bookings": bookings,
    }

@app.get("/movies/summary")
def get_movies_summary():
    if not movies:
        return {"total_movies": 0}

    genre_count: dict = {}
    for m in movies:
        genre_count[m["genre"]] = genre_count.get(m["genre"], 0) + 1

    most_expensive = max(movies, key=lambda m: m["ticket_price"])
    cheapest       = min(movies, key=lambda m: m["ticket_price"])
    total_seats    = sum(m["seats_available"] for m in movies)

    return {
        "total_movies": len(movies),
        "most_expensive_ticket": {"title": most_expensive["title"], "price": most_expensive["ticket_price"]},
        "cheapest_ticket":       {"title": cheapest["title"],       "price": cheapest["ticket_price"]},
        "total_seats_available": total_seats,
        "movies_by_genre":       genre_count,
    }

@app.get("/movies/search")
def search_movies(keyword: str = Query(..., min_length=1)):
    kw = keyword.lower()
    matches = [
        m for m in movies
        if kw in m["title"].lower()
        or kw in m["genre"].lower()
        or kw in m["language"].lower()
    ]
    if not matches:
        return {"total_found": 0, "message": "No movies found matching your search.", "movies": []}
    return {"total_found": len(matches), "movies": matches}

VALID_MOVIE_SORT = {"ticket_price", "title", "duration_mins", "seats_available"}


@app.get("/movies/sort")
def sort_movies(sort_by: str = "ticket_price", order: str = "asc"):
    if sort_by not in VALID_MOVIE_SORT:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sort_by. Choose from: {', '.join(VALID_MOVIE_SORT)}",
        )
    if order not in ("asc", "desc"):
        raise HTTPException(status_code=400, detail="order must be 'asc' or 'desc'")
    sorted_movies = sorted(movies, key=lambda m: m[sort_by], reverse=(order == "desc"))
    return {"sort_by": sort_by, "order": order, "movies": sorted_movies}

@app.get("/movies/page")
def paginate_movies(
    page:  int = Query(default=1, ge=1),
    limit: int = Query(default=3, ge=1),
):
    total       = len(movies)
    total_pages = math.ceil(total / limit) if total else 1
    start       = (page - 1) * limit
    return {
        "page":        page,
        "limit":       limit,
        "total":       total,
        "total_pages": total_pages,
        "movies":      movies[start:start + limit],
    }

def filter_movies_logic(
    movie_list,
    genre:     Optional[str] = None,
    language:  Optional[str] = None,
    max_price: Optional[int] = None,
    min_seats: Optional[int] = None,
):
    result = movie_list
    if genre     is not None:
        result = [m for m in result if m["genre"].lower()    == genre.lower()]
    if language  is not None:
        result = [m for m in result if m["language"].lower() == language.lower()]
    if max_price is not None:
        result = [m for m in result if m["ticket_price"]     <= max_price]
    if min_seats is not None:
        result = [m for m in result if m["seats_available"]  >= min_seats]
    return result


@app.get("/movies/filter")
def filter_movies(
    genre:     Optional[str] = None,
    language:  Optional[str] = None,
    max_price: Optional[int] = None,
    min_seats: Optional[int] = None,
):
    result = filter_movies_logic(movies, genre, language, max_price, min_seats)
    return {"total": len(result), "movies": result}

@app.get("/movies/browse")
def browse_movies(
    keyword:  Optional[str] = None,
    genre:    Optional[str] = None,
    language: Optional[str] = None,
    sort_by:  str = "ticket_price",
    order:    str = "asc",
    page:     int = Query(default=1, ge=1),
    limit:    int = Query(default=3, ge=1),
):
    if sort_by not in VALID_MOVIE_SORT:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sort_by. Choose from: {', '.join(VALID_MOVIE_SORT)}",
        )
    if order not in ("asc", "desc"):
        raise HTTPException(status_code=400, detail="order must be 'asc' or 'desc'")

    result = list(movies)

    if keyword:
        kw = keyword.lower()
        result = [
            m for m in result
            if kw in m["title"].lower()
            or kw in m["genre"].lower()
            or kw in m["language"].lower()
        ]

    result = filter_movies_logic(result, genre=genre, language=language)

    result = sorted(result, key=lambda m: m[sort_by], reverse=(order == "desc"))

    total       = len(result)
    total_pages = math.ceil(total / limit) if total else 1
    start       = (page - 1) * limit

    return {
        "page":           page,
        "limit":          limit,
        "total_matched":  total,
        "total_pages":    total_pages,
        "filters_applied": {"keyword": keyword, "genre": genre, "language": language},
        "sort_by":        sort_by,
        "order":          order,
        "movies":         result[start:start + limit],
    }

def find_movie(movie_id: int):
    """Q7 helper — find a movie by id or return None."""
    for m in movies:
        if m["id"] == movie_id:
            return m
    return None


@app.get("/movies/{movie_id}")
def get_movie(movie_id: int):
    movie = find_movie(movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail=f"Movie with id {movie_id} not found")
    return movie

def calculate_ticket_cost(base_price: int, seats: int, seat_type: str, promo_code: str = ""):
    """
    Q7 + Q9:
      standard  → 1×  base price
      premium   → 1.5× base price
      recliner  → 2×  base price
    Promo codes: SAVE10 → 10% off | SAVE20 → 20% off
    """
    multipliers = {"standard": 1.0, "premium": 1.5, "recliner": 2.0}
    multiplier  = multipliers.get(seat_type.lower(), 1.0)

    original_cost = round(base_price * multiplier * seats)

    discount_pct = 0
    if promo_code.upper() == "SAVE10":
        discount_pct = 10
    elif promo_code.upper() == "SAVE20":
        discount_pct = 20

    discount_amount = round(original_cost * discount_pct / 100)
    total_cost      = original_cost - discount_amount

    return {
        "original_cost":    original_cost,
        "promo_code":       promo_code.upper() if promo_code else None,
        "discount_percent": discount_pct,
        "discount_amount":  discount_amount,
        "total_cost":       total_cost,
    }

class BookingRequest(BaseModel):
    customer_name: str  = Field(..., min_length=2)
    movie_id:      int  = Field(..., gt=0)
    seats:         int  = Field(..., gt=0, le=10)
    phone:         str  = Field(..., min_length=10)
    seat_type:     str  = "standard"
    promo_code:    str  = ""         

@app.post("/bookings", status_code=201)
def create_booking(req: BookingRequest):
    global booking_counter

    movie = find_movie(req.movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail=f"Movie with id {req.movie_id} not found")
    if movie["seats_available"] < req.seats:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough seats. Requested: {req.seats}, Available: {movie['seats_available']}",
        )

    cost = calculate_ticket_cost(movie["ticket_price"], req.seats, req.seat_type, req.promo_code)

    booking = {
        "booking_id":   booking_counter,
        "customer_name": req.customer_name,
        "phone":        req.phone,
        "movie_id":     movie["id"],
        "movie_title":  movie["title"],
        "seats":        req.seats,
        "seat_type":    req.seat_type,
        "status":       "confirmed",
        **cost,
    }

    movie["seats_available"] -= req.seats
    bookings.append(booking)
    booking_counter += 1
    return booking

class NewMovie(BaseModel):
    title:          str = Field(..., min_length=2)
    genre:          str = Field(..., min_length=2)
    language:       str = Field(..., min_length=2)
    duration_mins:  int = Field(..., gt=0)
    ticket_price:   int = Field(..., gt=0)
    seats_available: int = Field(..., gt=0)


@app.post("/movies", status_code=201)
def add_movie(new_movie: NewMovie):
    global movie_id_counter

    for m in movies:
        if m["title"].lower() == new_movie.title.lower():
            raise HTTPException(
                status_code=400,
                detail=f"Movie '{new_movie.title}' already exists",
            )

    movie_dict = {
        "id":             movie_id_counter,
        "title":          new_movie.title,
        "genre":          new_movie.genre,
        "language":       new_movie.language,
        "duration_mins":  new_movie.duration_mins,
        "ticket_price":   new_movie.ticket_price,
        "seats_available": new_movie.seats_available,
    }
    movies.append(movie_dict)
    movie_id_counter += 1
    return movie_dict

@app.put("/movies/{movie_id}")
def update_movie(
    movie_id:        int,
    ticket_price:    Optional[int] = Query(default=None, gt=0),
    seats_available: Optional[int] = Query(default=None, ge=0),
):
    movie = find_movie(movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail=f"Movie with id {movie_id} not found")

    if ticket_price    is not None:
        movie["ticket_price"]    = ticket_price
    if seats_available is not None:
        movie["seats_available"] = seats_available

    return {"message": "Movie updated successfully", "movie": movie}

@app.delete("/movies/{movie_id}")
def delete_movie(movie_id: int):
    movie = find_movie(movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail=f"Movie with id {movie_id} not found")

    existing = [b for b in bookings if b["movie_id"] == movie_id]
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete movie '{movie['title']}' — it has {len(existing)} existing booking(s)",
        )

    movies.remove(movie)
    return {"message": f"Movie '{movie['title']}' deleted successfully"}

holds = []
hold_counter = 1


class SeatHoldRequest(BaseModel):
    customer_name: str = Field(..., min_length=2)
    movie_id:      int = Field(..., gt=0)
    seats:         int = Field(..., gt=0, le=10)


@app.post("/seat-hold", status_code=201)
def create_seat_hold(req: SeatHoldRequest):
    global hold_counter

    movie = find_movie(req.movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail=f"Movie with id {req.movie_id} not found")
    if movie["seats_available"] < req.seats:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough seats to hold. Requested: {req.seats}, Available: {movie['seats_available']}",
        )

    hold = {
        "hold_id":       hold_counter,
        "customer_name": req.customer_name,
        "movie_id":      movie["id"],
        "movie_title":   movie["title"],
        "seats":         req.seats,
        "status":        "on_hold",
    }

    movie["seats_available"] -= req.seats
    holds.append(hold)
    hold_counter += 1
    return hold


@app.get("/seat-hold")
def get_all_holds():
    return {"total": len(holds), "holds": holds}

def find_hold(hold_id: int):
    for h in holds:
        if h["hold_id"] == hold_id:
            return h
    return None


@app.post("/seat-confirm/{hold_id}", status_code=201)
def confirm_seat_hold(hold_id: int):
    global booking_counter

    hold = find_hold(hold_id)
    if not hold:
        raise HTTPException(status_code=404, detail=f"Hold with id {hold_id} not found")

    movie = find_movie(hold["movie_id"])
    cost  = calculate_ticket_cost(
        movie["ticket_price"] if movie else 0,
        hold["seats"],
        "standard",
    )

    booking = {
        "booking_id":    booking_counter,
        "customer_name": hold["customer_name"],
        "phone":         "N/A",
        "movie_id":      hold["movie_id"],
        "movie_title":   hold["movie_title"],
        "seats":         hold["seats"],
        "seat_type":     "standard",
        "status":        "confirmed",
        **cost,
    }

    bookings.append(booking)
    holds.remove(hold)
    booking_counter += 1
    return {"message": "Seat hold confirmed and booking created", "booking": booking}


@app.delete("/seat-release/{hold_id}")
def release_seat_hold(hold_id: int):
    hold = find_hold(hold_id)
    if not hold:
        raise HTTPException(status_code=404, detail=f"Hold with id {hold_id} not found")

    movie = find_movie(hold["movie_id"])
    if movie:
        movie["seats_available"] += hold["seats"]

    holds.remove(hold)
    return {
        "message": f"Hold {hold_id} released. {hold['seats']} seat(s) returned to '{hold['movie_title']}'"
    }

VALID_BOOKING_SORT = {"total_cost", "seats"}


@app.get("/bookings/search")
def search_bookings(customer_name: str = Query(..., min_length=1)):
    kw      = customer_name.lower()
    matches = [b for b in bookings if kw in b["customer_name"].lower()]
    return {"total_found": len(matches), "bookings": matches}


@app.get("/bookings/sort")
def sort_bookings(sort_by: str = "total_cost", order: str = "asc"):
    if sort_by not in VALID_BOOKING_SORT:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sort_by. Choose from: {', '.join(VALID_BOOKING_SORT)}",
        )
    if order not in ("asc", "desc"):
        raise HTTPException(status_code=400, detail="order must be 'asc' or 'desc'")
    sorted_bookings = sorted(bookings, key=lambda b: b[sort_by], reverse=(order == "desc"))
    return {"sort_by": sort_by, "order": order, "bookings": sorted_bookings}


@app.get("/bookings/page")
def paginate_bookings(
    page:  int = Query(default=1, ge=1),
    limit: int = Query(default=3, ge=1),
):
    total       = len(bookings)
    total_pages = math.ceil(total / limit) if total else 1
    start       = (page - 1) * limit
    return {
        "page":        page,
        "limit":       limit,
        "total":       total,
        "total_pages": total_pages,
        "bookings":    bookings[start:start + limit],
    }
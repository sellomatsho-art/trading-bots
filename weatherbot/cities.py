"""City registry for temperature markets.

Coordinates are the official observation station Polymarket resolves against
where one is documented (typically the NWS/ASOS airport site), because a
downtown lat/lon can run several degrees warmer than the airport that actually
settles the market.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class City:
    key: str
    name: str
    latitude: float
    longitude: float
    timezone: str
    station: str
    #: Lowercase substrings that identify this city in a market question.
    aliases: tuple[str, ...] = field(default_factory=tuple)

    def matches(self, text: str) -> bool:
        low = text.lower()
        return any(alias in low for alias in self.aliases)


CITIES: dict[str, City] = {
    c.key: c
    for c in [
        City(
            key="nyc",
            name="New York City",
            latitude=40.7794,
            longitude=-73.8803,
            timezone="America/New_York",
            station="KLGA (LaGuardia)",
            aliases=("nyc", "new york"),
        ),
        City(
            key="chicago",
            name="Chicago",
            latitude=41.9803,
            longitude=-87.9090,
            timezone="America/Chicago",
            station="KORD (O'Hare)",
            aliases=("chicago",),
        ),
        City(
            key="la",
            name="Los Angeles",
            latitude=33.9382,
            longitude=-118.3866,
            timezone="America/Los_Angeles",
            station="KLAX",
            aliases=("los angeles", "la ", "lax"),
        ),
        City(
            key="miami",
            name="Miami",
            latitude=25.7932,
            longitude=-80.2906,
            timezone="America/New_York",
            station="KMIA",
            aliases=("miami",),
        ),
        City(
            key="austin",
            name="Austin",
            latitude=30.1975,
            longitude=-97.6664,
            timezone="America/Chicago",
            station="KAUS",
            aliases=("austin",),
        ),
        City(
            key="denver",
            name="Denver",
            latitude=39.8467,
            longitude=-104.6562,
            timezone="America/Denver",
            station="KDEN",
            aliases=("denver",),
        ),
        City(
            key="london",
            name="London",
            latitude=51.4790,
            longitude=-0.4493,
            timezone="Europe/London",
            station="EGLL (Heathrow)",
            aliases=("london",),
        ),
        City(
            key="paris",
            name="Paris",
            latitude=49.0097,
            longitude=2.5479,
            timezone="Europe/Paris",
            station="LFPG (Charles de Gaulle)",
            aliases=("paris",),
        ),
        City(
            key="tokyo",
            name="Tokyo",
            latitude=35.5533,
            longitude=139.7811,
            timezone="Asia/Tokyo",
            station="RJTT (Haneda)",
            aliases=("tokyo",),
        ),
        City(
            key="seoul",
            name="Seoul",
            latitude=37.5665,
            longitude=126.9780,
            timezone="Asia/Seoul",
            station="RKSS (Gimpo)",
            aliases=("seoul",),
        ),
    ]
}


def find_city(text: str) -> City | None:
    """Return the city named in a market question, or None."""
    for city in CITIES.values():
        if city.matches(text):
            return city
    return None

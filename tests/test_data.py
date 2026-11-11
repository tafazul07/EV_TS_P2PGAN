from src.data_loader import read_trip


def test_windows1252_and_tripb38_header_alias():
    frame = read_trip("data/extracted/TripB38.csv")
    assert "Battery Temperature [°C]" in frame.columns
    assert "Velocity [km/h]" in frame.columns
    assert frame.attrs["header_aliases"] == {"Velocity [km/h]]]": "Velocity [km/h]"}


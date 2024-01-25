import glob


def get_season_for_month(month: int) -> str:
    """
    This method get the season for a specific month for a number of a month.

    @param month: A month in number
    @return the season in string format, and the season in string format.
    """

    season = int(month) % 12 // 3 + 1
    season_str = ""
    if season == 1:
        season_str = "Winter"
    if season == 2:
        season_str = "Spring"
    if season == 3:
        season_str = "Summer"
    if season == 4:
        season_str = "Fall"

    return season_str


def get_scaler_filepath(folder: str, image_date: str, location: str, band: int) -> str:
    regex = f"{folder}/{image_date}*{location}*band{band}*"
    filepath = glob.glob(regex)[0].replace("\\", "/")
    return filepath

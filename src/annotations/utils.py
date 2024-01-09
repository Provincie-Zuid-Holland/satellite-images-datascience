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

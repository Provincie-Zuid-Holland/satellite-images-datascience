from enum import Enum


class FeatureType(Enum):
    """
    Shows which types of features are available
    """

    FRACTION_BRIGHT = "fraction_bright"
    FRACTION_RELATIVE_BRIGHT = "fraction_relative_bright"
    FRACTION_COLOUR_BRIGHT = "fraction_colour_bright"
    COLOUR_MEAN = "colour_mean"
    COLOUR_QUANTILE = "colour_quantile"
    NUMBER_BRIGHT_PIXELS = "number_bright_pixels"
    NUMBER_OF_PIXELS = "number_of_pixels"
    FRACTION_BRIGHT_FROM_MAX = "fraction_bright_from_max"


class Feature:
    """
    Shows which features with specific properties are available.

    i.e.
    feature_type = FeatureType.FRACTION_COLOUR_BRIGHT
    colour = 'red'
    minimal_brightness = 500

    corresponds to column name 'fraction_red_bright_500'
    """

    def __init__(self, feature_type: FeatureType, **kwargs):
        self.feature_type = feature_type
        self.kwargs = kwargs

    def __str__(self):
        representation = self.feature_type.value
        if "colour" in self.kwargs.keys():
            representation = representation.replace("colour", self.kwargs["colour"])
        for key in self.kwargs.keys():
            if key != "colour":
                representation = f"{representation}_{self.kwargs[key]}"

        return representation


all_minimal_brightnesses = [500, 600, 700, 800, 900, 1000, 1100, 1200]
all_minimal_fraction_brightnesses = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
all_minimal_quantiles = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
all_colours = ["red", "green", "blue"]
all_colour_quantiles = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]

# All features that have been tried out during model development
all_features = (
    [
        Feature(
            feature_type=FeatureType.FRACTION_BRIGHT,
            minimal_brightness=minimal_brightness,
        )
        for minimal_brightness in all_minimal_brightnesses
    ]
    + [
        Feature(
            feature_type=FeatureType.FRACTION_RELATIVE_BRIGHT,
            minimal_quantile=minimal_quantile,
        )
        for minimal_quantile in all_minimal_quantiles
    ]
    + [
        Feature(
            feature_type=FeatureType.FRACTION_COLOUR_BRIGHT,
            colour=colour,
            minimal_brightness=minimal_brightness,
        )
        for minimal_brightness in all_minimal_brightnesses
        for colour in all_colours
    ]
    + [
        Feature(feature_type=FeatureType.COLOUR_MEAN, colour=colour)
        for colour in all_colours
    ]
    + [
        Feature(
            feature_type=FeatureType.COLOUR_QUANTILE, colour=colour, quantile=quantile
        )
        for quantile in all_colour_quantiles
        for colour in all_colours
    ]
    + [
        Feature(
            feature_type=FeatureType.NUMBER_BRIGHT_PIXELS,
            minimal_brightness=minimal_brightness,
        )
        for minimal_brightness in all_minimal_brightnesses
    ]
    + [Feature(feature_type=FeatureType.NUMBER_OF_PIXELS)]
    + [
        Feature(
            feature_type=FeatureType.FRACTION_BRIGHT_FROM_MAX,
            minimal_fraction_brightness=minimal__fraction_brightness,
        )
        for minimal__fraction_brightness in all_minimal_fraction_brightnesses
    ]
)

# The features deemed most effective in predicting cloud_detection
selected_features = [
    Feature(feature_type=FeatureType.FRACTION_BRIGHT, minimal_brightness=500),
    Feature(feature_type=FeatureType.FRACTION_BRIGHT, minimal_brightness=700),
    Feature(feature_type=FeatureType.FRACTION_RELATIVE_BRIGHT, minimal_quantile=0.6),
    Feature(feature_type=FeatureType.FRACTION_RELATIVE_BRIGHT, minimal_quantile=0.8),
    Feature(feature_type=FeatureType.FRACTION_RELATIVE_BRIGHT, minimal_quantile=0.9),
    Feature(feature_type=FeatureType.FRACTION_RELATIVE_BRIGHT, minimal_quantile=0.95),
    Feature(feature_type=FeatureType.FRACTION_RELATIVE_BRIGHT, minimal_quantile=0.99),
    Feature(
        feature_type=FeatureType.FRACTION_COLOUR_BRIGHT,
        colour="green",
        minimal_brightness=500,
    ),
    Feature(feature_type=FeatureType.COLOUR_QUANTILE, colour="red", quantile=0.99),
    Feature(feature_type=FeatureType.NUMBER_BRIGHT_PIXELS, minimal_brightness=500),
    Feature(feature_type=FeatureType.NUMBER_BRIGHT_PIXELS, minimal_brightness=700),
    Feature(feature_type=FeatureType.NUMBER_OF_PIXELS),
]

import satellite_images_nso_datascience.model_metrics.custom_model_metrics as custom_model_metrics
import pickle
import tests.annotation_model_tests.test_settings as test_settings


def test_superview_model():

    final_model_filepath = test_settings.model_path_sv
    final_artefact = pickle.load(open(final_model_filepath, "rb"))

    nk_plassen_metrics = custom_model_metrics.custom_model_metrics(
        final_artefact["model"],
        final_artefact["scaler"],
        "Nieuwkoopse_plassen",
        "Superview",
    )

    assert False not in nk_plassen_metrics.metrics_on_small_tif_file_iterator()


def test_pneo_model():

    final_model_filepath = test_settings.model_path_pneo
    final_artefact = pickle.load(open(final_model_filepath, "rb"))

    nk_plassen_metrics = custom_model_metrics.custom_model_metrics(
        final_artefact["model"],
        final_artefact["scaler"],
        "Nieuwkoopse_plassen",
        "PNEO",
    )

    assert False not in nk_plassen_metrics.metrics_on_small_tif_file_iterator()

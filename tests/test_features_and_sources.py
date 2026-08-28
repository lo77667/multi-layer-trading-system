from trading_system.data import generate_sample_candles
from trading_system.features import engineer_features, make_direction_labels, model_matrix
from trading_system.public_api_sources import PUBLIC_API_SOURCES


def test_advanced_features_are_created_without_future_shift():
    candles = generate_sample_candles(160)
    frame = engineer_features(candles)
    expected = {"rsi_7", "rsi_14", "rsi_21", "macd_histogram", "bollinger_width", "keltner_width", "volume_ratio_20", "atr_14"}
    assert expected.issubset(frame.columns)
    labels = make_direction_labels(frame, horizon=8)
    assert labels.iloc[-8:].isna().all()
    matrix, columns = model_matrix(frame)
    assert "close" not in columns
    assert len(matrix) < len(frame)


def test_public_api_catalog_contains_selected_sources():
    names = {source.name for source in PUBLIC_API_SOURCES}
    assert {"Twelve Data", "Frankfurter", "GNews"}.issubset(names)

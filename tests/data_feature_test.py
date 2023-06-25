import pandas
import pytest


@pytest.fixture()
def test_data():
    test_file = "./data/raw/a1_RestaurantReviews_HistoricDump.tsv"
    data = pandas.read_csv(test_file, sep="\t", quoting=3)
    yield data


def test_distribution(test_data):
    positive = len(test_data[test_data["Liked"] == 1].reset_index())
    negative = len(test_data[test_data["Liked"] == 0].reset_index())
    assert abs(1 - positive / negative) <= 0.3

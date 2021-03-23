import numpy as np
import pandas as pd

from cancer_package.preprocessing import BasicPreprocessing
from cancer_package import constants


mock_data = pd.DataFrame(
        {
            "patient_id": [1, 2, 3, 4, 5],
            "A": [1, -1, 9, 2, .6],
            "B": [-1, -1, 8, -1, 7],
            "C": [4, -1, -1, -1, 7],
            "category": ["a", "a", "b", "b", "b"]
        }
    )


def test_data_prep_init():
    data_prep = BasicPreprocessing(mock_data, -1)
    assert data_prep.proteins == ["A", "B", "C"]
    pd.testing.assert_frame_equal(
        data_prep.data,
        mock_data[["patient_id", "category", "A", "B", "C"]]
    )


def test_rm_no_groups():
    data_prep = BasicPreprocessing(mock_data, -1)
    data_prep.rm_execess_nans(.7, by_group=False)
    assert data_prep.proteins == ["A"]
    pd.testing.assert_frame_equal(
        data_prep.data,
        mock_data[["patient_id", "category", "A"]]
    )


def test_rm_with_groups_high_threshold():
    data_prep = BasicPreprocessing(mock_data, -1)
    data_prep.rm_execess_nans(.99)
    assert data_prep.proteins == ["A"]
    pd.testing.assert_frame_equal(
        data_prep.data,
        mock_data[["patient_id", "category", "A"]]
    )


def test_rm_with_groups_low_threshold():
    data_prep = BasicPreprocessing(mock_data, -1)
    data_prep.rm_execess_nans(.6)
    assert data_prep.proteins == ["A", "B"]
    pd.testing.assert_frame_equal(
        data_prep.data,
        mock_data[["patient_id", "category", "A", "B"]]
    )


def test_rm_high_threshold():
    data_prep = BasicPreprocessing(mock_data, -1)
    data_prep.rm_execess_nans(.99, by_group=False)
    assert data_prep.proteins == []
    pd.testing.assert_frame_equal(
        data_prep.data,
        mock_data[["patient_id", "category"]]
    )


def test_rm_low_threshold():
    data_prep = BasicPreprocessing(mock_data, -1)
    data_prep.rm_execess_nans(.6, by_group=False)
    assert data_prep.proteins == ["A"]
    pd.testing.assert_frame_equal(
        data_prep.data,
        mock_data[["patient_id", "category", "A"]]
    )


def test_rm_energy_proteins():
    mock_data2 = mock_data.rename(columns={"A": 'CAT'})
    data_prep = BasicPreprocessing(mock_data2, -1, use_energy_proteins=False)
    assert data_prep.proteins == ["B", "C"]
    pd.testing.assert_frame_equal(
        data_prep.data,
        mock_data2[["patient_id", "category", "B", "C"]]
    )


def test_organise_proteins_all_present():
    data_prep = BasicPreprocessing(mock_data, -1)
    protein_group = pd.DataFrame({"Protein": ["B", "C", "A"]})
    data_prep.organise_proteins(protein_group)
    assert data_prep.proteins == ["B", "C", "A"]
    pd.testing.assert_frame_equal(
        data_prep.data,
        mock_data[["patient_id", "category", "B", "C", "A"]]
    )


def test_organise_proteins_with_missing():
    mock_data2 = mock_data.copy()
    mock_data2["D"] = np.arange(len(mock_data2))
    data_prep = BasicPreprocessing(mock_data2, -1)
    protein_group = pd.DataFrame({"Protein": ["C", "B"]})
    data_prep.organise_proteins(protein_group)
    assert data_prep.proteins == ["C", "B", "A", "D"]
    pd.testing.assert_frame_equal(
        data_prep.data,
        mock_data2[["patient_id", "category", "C", "B", "A", "D"]]
    )



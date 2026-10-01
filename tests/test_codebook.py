from alzheimers_trajectory_mapping.codebook import normalize_name, parse_codes


def test_normalize_name():
    assert normalize_name("A1_EventDateX") == "A1_EVENTDATEX"
    assert normalize_name("Proteomics_IL-12 P40 (HUMAN)") == "PROTEO_IL_12_P40_HUMAN"
    assert normalize_name("Proteomics_VEGF- ANGIO PLATE") == "PROTEO_VEGF_ANGIO_PLATE"
    assert normalize_name("C1_C1DATX") == "C1_DATEX"


def test_missing_codes():
    assert parse_codes("0-30, 99 = Missing", None)["missing"] == [99.0]
    # 88 = None reported is a real answer, not missing
    codes = parse_codes("1 = White, 88 = None reported, 99 = Unknown", None)
    assert codes["missing"] == [99.0]


def test_not_applicable_kept_separate():
    codes = parse_codes("0-10, 88 = NA, 99 = Unknown", None)
    assert codes == {"missing": [99.0], "not_applicable": [88.0],
                     "below_detection": [], "above_detection": []}


def test_detection_limits_not_missing():
    text = ("-999 = No sample; -555 = Value above highest detectable limit of that assay; "
            "-333 = Value below lowest detectable limit of that assay")
    codes = parse_codes(text, "-999, -555, -333")
    assert codes["missing"] == [-999.0]
    assert codes["below_detection"] == [-333.0]
    assert codes["above_detection"] == [-555.0]
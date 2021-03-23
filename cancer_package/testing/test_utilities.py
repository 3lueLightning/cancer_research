from cancer_package import constants
from cancer_package.utilities import multiple_replace


def test_string_simplification_no_replacement():
    orginal_str = 'HSP90 / ST13'
    new_str = multiple_replace(orginal_str, constants.EQUATION_SIMPLIFIER)
    assert new_str == 'HSP90 / ST13', "the new string isn't as it should be"


def test_string_simplification_1_replacement():
    orginal_str = '(HSPA2 + HSPA6 + HSPA8 + HSPA5 + HSPA9 + HSPA12A) / ST13'
    new_str = multiple_replace(orginal_str, constants.EQUATION_SIMPLIFIER)
    assert new_str == 'HSP70 / ST13', "the new string isn't as it should be"


def test_string_simplification_2_replacements():
    numerator = "(DNAJA1 + DNAJA2 + DNAJC11 + DNAJB1 + DNAJC5 + DNAJC13)"
    denominator = "(HSPA2 + HSPA6 + HSPA8 + HSPA5 + HSPA9 + HSPA12A)"
    orginal_str = f"{numerator} / {denominator}"
    new_str = multiple_replace(orginal_str, constants.EQUATION_SIMPLIFIER)
    assert new_str == 'DNAJ / HSP70', "the new string isn't as it should be"
    
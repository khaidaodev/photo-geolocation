"""
Groups the country codes into continents, so the later analysis can ask whether the mistakes
stay on the right continent even when the country is wrong.

The codes are the two letter ISO ones that the Country211 folders are named with. Some
countries sit on a border, so the choice is mine: Russia, Turkey, Cyprus, Georgia, Armenia,
Azerbaijan and Kazakhstan are the ones that could go either way. Check them against
src/check_continents.py before trusting any number built on this.
"""

CONTINENTS = {
    "Africa": (
        "AO BF BI BJ BW CD CF CG CI CM CV DJ DZ EG EH ER ET GA GH GM GN GQ GW KE KM LR LS LY "
        "MA MG ML MR MU MW MZ NA NE NG RE RW SC SD SH SL SN SO SS ST SZ TD TG TN TZ UG YT ZA "
        "ZM ZW"
    ),
    "Asia": (
        "AE AF AM AZ BD BH BN BT CC CN CX CY GE HK ID IL IN IO IQ IR JO JP KG KH KP KR KW KZ "
        "LA LB LK MM MN MO MV MY NP OM PH PK PS QA SA SG SY TH TJ TL TM TR TW UZ VN YE"
    ),
    "Europe": (
        "AD AL AT AX BA BE BG BY CH CZ DE DK EE ES FI FO FR GB GG GI GR HR HU IE IM IS IT JE "
        "LI LT LU LV MC MD ME MK MT NL NO PL PT RO RS RU SE SI SJ SK SM UA VA XK"
    ),
    "North America": (
        "AG AI AW BB BL BM BQ BS BZ CA CR CU CW DM DO GD GL GP GT HN HT JM KN KY LC MF MQ MS "
        "MX NI PA PM PR SV SX TC TT UM US VC VG VI"
    ),
    "South America": "AR BO BR CL CO EC FK GF GY PE PY SR UY VE",
    "Oceania": (
        "AS AU CK FJ FM GU KI MH MP NC NF NR NU NZ PF PG PN PW SB TK TO TV VU WF WS"
    ),
    "Antarctica": "AQ BV GS HM TF",
}

CONTINENT_OF = {
    code: continent for continent, codes in CONTINENTS.items() for code in codes.split()
}


def continent_of(code):
    """Return the continent for a two letter country code. Raises KeyError if it's not known."""
    return CONTINENT_OF[code.upper()]

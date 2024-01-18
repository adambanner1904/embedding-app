import seaborn as sns


def get_colour_key():
    religion = ("alt.atheism",
                "talk.religion.misc",
                "soc.religion.christian")
    politics = ("talk.politics.misc",
                "talk.politics.mideast",
                "talk.politics.guns")
    sport = ("rec.sport.baseball",
             "rec.sport.hockey")
    comp = (
        "comp.graphics",
        "comp.os.ms-windows.misc",
        "comp.sys.ibm.pc.hardware",
        "comp.sys.mac.hardware",
        "comp.windows.x",
    )
    sci = (
        "sci.crypt",
        "sci.electronics",
        "sci.med",
        "sci.space",
    )
    misc = (
        "misc.forsale",
        "rec.autos",
        "rec.motorcycles",
    )

    COLOR_KEY = {}
    COLOR_KEY.update(zip(religion, sns.color_palette("Blues", 4).as_hex()[1:]))
    COLOR_KEY.update(zip(politics, sns.color_palette("Purples", 4).as_hex()[1:]))
    COLOR_KEY.update(zip(comp, sns.color_palette("YlOrRd", 5).as_hex()))
    COLOR_KEY.update(zip(sci, sns.color_palette("light:teal", 5).as_hex()[1:]))
    COLOR_KEY.update(zip(sport, sns.color_palette("light:#660033", 4).as_hex()[1:3]))
    COLOR_KEY.update(zip(misc, sns.color_palette("YlGn", 4).as_hex()[1:]))
    return COLOR_KEY

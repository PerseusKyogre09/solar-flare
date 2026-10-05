import drms

client = drms.Client()

query = 'hmi.sharp_720s[][2010.05.05_00:00:00_TAI-2010.05.05_02:00:00_TAI][? NOAA_ARS ~ "11069" ?]'

result = client.query(
    query,
    key=[
        "HARPNUM",
        "T_REC",
        "NOAA_ARS",
        "USFLUX",
        "MEANGAM",
        "MEANGBT",
        "MEANGBZ",
        "MEANGBH",
        "MEANJZD",
        "TOTUSJZ",
        "MEANALP",
        "MEANJZH",
        "TOTUSJH",
        "ABSNJZH",
        "SAVNCPP",
        "MEANPOT",
        "TOTPOT",
        "MEANSHR",
        "SHRGT45",
        "R_VALUE",
        "AREA_ACR",
    ]
)

print(result.to_string(index=False))

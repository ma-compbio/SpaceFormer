BRAIN_FILENAMES = [
    "H18.06.006.MTG.4000.expand.rep1",
    "H18.06.006.MTG.4000.expand.rep3",
    "H22.26.401.MTG.4000.expand.rep1",
    "H22.26.401.MTG.4000.expand.rep2",
    "H19.30.001.STG.4000.expand.rep1",
    "H19.30.001.STG.4000.expand.rep2",
    "H20.30.001.STG.4000.expand.rep2",
    "H20.30.001.STG.4000.expand.rep3",
    "H20.30.001.STG.4000.expand.rep1",
    "H18.06.006.MTG.4000.expand.rep2",
]

BRAIN_LABEL_DICT = {
    'oENDO': 0,
    'lMGC': 1,
    'lOGC': 2,
    'lASC': 3,
    'eL2/3.IT': 4,
    'lOPC': 5,
    'oMURAL': 6,
    'eL5.IT': 7,
    'eL4/5.IT': 8,
    'iVIP': 9,
    'iSST': 10,
    'iPVALB': 11,
    'iLAMP5': 12,
    'eL5/6.NP': 13,
    'eL6.IT.CAR3': 14,
    'eL6.IT': 15,
    'eL5.ET': 16,
    'eL6b': 17,
    'eL6.CT': 18,
    'oVLMC': 19,
    'eNP': 20,
    'eL6.CAR3': 21,
    'iSNCG': 22,
    'iMEIS2': 23
}


GSEA_LABEL_DICT = {
    'Astro-1': 0,
    'Astro-2': 1,
    'Endo-1': 2,
    'Endo-2': 3,
    'Endo-3': 4,
    'Epen': 5,
    'ExN-L2/3-1': 6,
    'ExN-L2/3-2': 7,
    'ExN-L5-1': 8,
    'ExN-L5-2': 9,
    'ExN-L5-3': 10,
    'ExN-L6-1': 11,
    'ExN-L6-2': 12,
    'ExN-L6-3': 13,
    'ExN-Olf': 14,
    'InN-Calb2-1': 15,
    'InN-Calb2-2': 16,
    'InN-Chat': 17,
    'InN-Lamp5': 18,
    'InN-Lhx6': 19,
    'InN-Olf-1': 20,
    'InN-Olf-2': 21,
    'InN-Pvalb-1': 22,
    'InN-Pvalb-2': 23,
    'InN-Pvalb-3': 24,
    'InN-Sst-1': 25,
    'InN-Sst-2': 26,
    'InN-Vip': 27,
    'MSN-D1-1': 28,
    'MSN-D1-2': 29,
    'MSN-D2': 30,
    'Macro': 31,
    'Micro-1': 32,
    'Micro-2': 33,
    'Micro-3': 34,
    'OPC': 35,
    'Olig-1': 36,
    'Olig-2': 37,
    'Olig-3': 38,
    'Peri-1': 39,
    'Peri-2': 40,
    'T cell': 41,
    'Vlmc': 42
}

GSEA_LABEL_DICT_REVERSE = {item: key for key, item in GSEA_LABEL_DICT.items()}

HUMAN_4000_FILENAMES = [
    "H18.06.006.MTG.4000.expand.rep1",
    "H18.06.006.MTG.4000.expand.rep2",
    "H18.06.006.MTG.4000.expand.rep3",
    "H22.26.401.MTG.4000.expand.rep1",
    "H22.26.401.MTG.4000.expand.rep2",
    "H19.30.001.STG.4000.expand.rep1",
    "H19.30.001.STG.4000.expand.rep2",
    "H20.30.001.STG.4000.expand.rep1",
    "H20.30.001.STG.4000.expand.rep2",
    "H20.30.001.STG.4000.expand.rep3"
]


MOUSE_FILENAMES = [
    "mouse1.AUD_TEA_VIS.242.unexpand",
    "mouse2.AUD_TEA_VIS.242.unexpand"
]
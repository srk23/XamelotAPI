# This file stores all the hardcoded parameters and variables needed for the cleaning process.

import re

from project.misc.dataframes import intersect_columns


# Columns with heterogeneous data type.
HETEROGENEOUS_COLUMNS = {'rhosp', 'mgrade'}

# Values generally referring to unknown values
GENERIC_UNKNOWNS = {888, 999}

# Sets of values standing for unknown values (column specific), 
SPECIFIC_UNKNOWNS = {
    # T, O
    'rhosp': {'nan'},
    'rsex': {'NR', 'UK', 'UKN'},
    'rbg': {9},
    'rethnic': {'Unknown', 'Not reported'},
    'rcmv': {3, 4, 6, 7, 8, 9},
    'rhcv': {3, 4, 6, 7, 8, 9},
    'rhbv': {3, 4, 6, 7, 8, 9},
    'rhiv': {3, 4, 6, 7, 8, 9},
    'dial_at_reg': {8, 9},
    'dial_at_tx': {8, 9},
    'dial_at_tx_type': {8, 9},
    'r_a_hom': {9},
    'r_b_hom': {9},
    'r_dr_hom': {9},
    'prd': {299},
    # T
    'dsex': {'NR', 'UK', 'UKN'},
    'dbg': {9},
    'dcod': {88, 90, 98, 99},
    'dethnic': {'Unknown', 'Not reported'},
    'dcmv': {3, 4, 6, 7, 8, 9},
    'dhbsag': {3, 4, 6, 7, 8, 9},
    'dhbcab': {3, 4, 6, 7, 8, 9},
    'dhcv': {3, 4, 6, 7, 8, 9},
    'dhiv': {3, 4, 6, 7, 8, 9},
    'debv': {3, 4, 6, 7, 8, 9},
    'dhtlv': {3, 4, 6, 7, 8, 9},
    'dpast_alcohol_abuse': {8, 9},
    'dpast_cardio_disease': {8, 9},
    'dpast_diabetes': {8, 9},
    'dpast_drug_abuse': {8, 9},
    'dpast_hypertension': {8, 9},
    'dpast_liver_disease': {8, 9},
    'dpast_smoker': {8, 9},
    'dpast_tumour': {8, 9},
    'dpast_uti': {8, 9},
    'd_a_hom': {9},
    'd_b_hom': {9},
    'd_dr_hom': {9},
    'amm': {7, 9},
    'bmm': {7, 9},
    'drmm': {7, 9},
    'aitx_type': {8},
    'mgrade': {'nan'},
    # O
    'reason1': {98, 99, 888},
    'reason2': {98, 99, 888},
    'rregcmv': {3, 4, 6, 7, 8, 9},
    'rreghcv': {3, 4, 6, 7, 8, 9},
    'rreghbv': {3, 4, 6, 7, 8, 9},
    'rreghiv': {3, 4, 6, 7, 8, 9},
    'rregdial_at_reg': {8, 9},
    'rreg_a_hom': {9},
    'rreg_b_hom': {9},
    'rreg_dr_hom': {9},
    'rregprd': {299}
}

# Sets of values standing for maximum values
LIMITS = [
    # https://en.wikipedia.org/wiki/List_of_heaviest_people (=636)
    ({'rweight', 'dweight', 'rregweight'}, (0, 250)),
    # https://en.wikipedia.org/wiki/List_of_tallest_people (=273)
    ({'rheight', 'dheight', 'rregheight'}, (0, 230)),
    # ALT <10000
    ({'alt_11', 'alt_12', 'alt_31', 'alt_32', 'alt_61',
      'alt_62', 'alt_71', 'alt_72', 'alt_73', 'alt_74',
      'alt_75', 'alt_81', 'alt_82', 'alt_83', 'alt_85'}, (-1, 10000)),
    # AST <10000
    ({'ast_11', 'ast_31', 'ast_61', 'ast_62', 'ast_71',
      'ast_72', 'ast_73', 'ast_74', 'ast_75', 'ast_81',
      'ast_82', 'ast_83'}, (-1, 10000)),
    # Amylase <2100
    ({'amylase_11', 'amylase_31', 'amylase_32', 'amylase_61',
      'amylase_62', 'amylase_71', 'amylase_72', 'amylase_73',
      'amylase_74', 'amylase_75', 'amylase_81', 'amylase_82',
      'amylase_85'}, (-1, 2100)),
    # Creatinine <1000
    ({'creatinine_11', 'creatinine_31', 'creatinine_61',
      'creatinine_62', 'creatinine_63', 'creatinine_71',
      'creatinine_72', 'creatinine_73', 'creatinine_74',
      'creatinine_75', 'creatinine_81', 'creatinine_82',
      'creatinine_83', 'creatinine_84'}, (-1, 1000)),
]

REFERENCES = [
    ({'rsex', 'dsex'},
     {
         'M': 'Male',
         'F': 'Female'
     }),
    ({'diab'},
     {
         0: 'No',
         1: 'Yes'
     }),
    ({
         'dial_at_tx', 'a_12', 'c_12', 'm_12', 'o_12', 'p_12', 't_12',
         'a_60', 'c_60', 'm_60', 'o_60', 'p_60', 't_60',
         'dpast_alcohol_abuse', 'dpast_cardio_disease', 'dpast_diabetes',
         'dpast_drug_abuse', 'dpast_hypertension', 'dpast_liver_disease',
         'dpast_smoker', 'dpast_tumour', 'dpast_uti'
     },
     {
         1: 'No',
         2: 'Yes'
     }),
    ({
         'dcmv', 'dhbsag', 'dhbcab', 'dhcv', 'dhiv', 'debv', 'dhtlv',
         'rregcmv', 'rreghcv', 'rreghbv', 'rreghiv',
         'rcmv', 'rhcv', 'rhbv', 'rhiv'
     },
     {
         1: 'Negative',
         2: 'Positive'
     }),
    ({'dial_at_reg', 'rregdial_at_reg', 'dial_at_tx_type'},
     {
         1: 'Haemodialysis',
         2: 'Peritoneal dialysis',
         3: 'Not on dialysis'
     }),
    ({'dtype'},
     {
         1: 'DBD',
         2: 'DCD'
     }),
    ({
         'r_a_hom', 'r_b_hom', 'r_dr_hom',
         'd_a_hom', 'd_b_hom', 'd_dr_hom',
         'rreg_a_hom', 'rreg_b_hom', 'rreg_dr_hom'
     },
     {
         0: 'Heterozygous',
         1: 'Homozygous'
     }),
    ({'tcens'},
     {
         0: 'Censored',
         1: 'Graft failure or death'
     }),
    ({'gcens'},
     {
         0: 'Censored',
         1: 'Graft failure'
     }),
    ({'pcens', 'rdeath'},
     {
         0: 'Censored',
         1: 'Death of recipient'
     }),
    ({'tx_type'},
     {
         10: 'Kidney',
         13: 'En-bloc kidney',
         14: 'Double kidney',
         110: 'Kidney & liver',
         115: 'Kidney, liver & pancreas',
         130: 'Kidney & pancreas',
         135: 'Kidney & pancreas islet',
         150: 'Kidney & heart',
         170: 'Kidney & lung',
         180: 'Kidney, heart & lung'
     }),
    ({'mgrade'},
     {
         'Non-fave': 'Non-favourable mismatch',
         'Fave': 'Favourable mismatch',
         '0': 'Zero mismatches'
     }),
    ({'status'},
     {
         1: 'Not offered',
         2: 'Not taken',
         3: 'Taken, but not accepted',
         4: 'Taken, accepted, but not used',
         5: 'Used'
     }),
    ({'hla_grp'},
     {
         1: 'Level 1',
         2: 'Level 2',
         3: 'Level 3',
         4: 'Level 4'
     }
     ),
    ({'dcod'},
     # "Looking at previous research, the most important predictors are usually CVA or trauma.
     #  Infection and malignancy are likely to reduce the chance of organ acceptance.
     #  Otherwise, COD is unlikely to affect decision making." (Simon)
     {
         10: 'Cerebrovascular',  # 'Intracranial haemorrhage',
         11: 'Cerebrovascular',  # 'Intracranial thrombosis',
         12: 'Malignancy',  # 'Brain tumour',
         13: 'Hypoxic brain damage - all causes',
         19: 'Cerebrovascular',  # 'Intracranial - type unclassified (CVA)',
         20: 'Trauma',  # 'Trauma RTA - car',
         21: 'Trauma',  # 'Trauma RTA - motorbike',
         22: 'Trauma',  # 'Trauma RTA - pushbike',
         23: 'Trauma',  # 'Trauma RTA - pedestrian',
         24: 'Trauma',  # 'Trauma RTA - Other',
         29: 'Trauma',  # 'Trauma RTA - unknown type',
         30: 'Known or suspected suicide',  # 'Other trauma - known or suspected suicide',
         31: 'Trauma',  # 'Other trauma - accident',
         39: 'Trauma',  # 'Other trauma - unknown cause',
         40: 'Cardiovascular',  # 'Cardiac arrest',
         41: 'Cardiovascular',  # 'Myocardial infarction',
         42: 'Cardiovascular',  # 'Aneurysm',
         43: 'Cardiovascular',  # 'Ischaemic heart disease',
         44: 'Cardiovascular',  # 'Congestive heart failure',
         45: 'Cardiovascular',  # 'Pulmonary embolism',
         49: 'Cardiovascular',  # 'Cardiovascular - type unclassified',
         50: 'Respiratory',  # 'Chronic pulmonary disease',
         51: 'Respiratory',  # 'Pneumonia',
         52: 'Respiratory',  # 'Asthma',
         53: 'Respiratory',  # 'Respiratory failure',
         54: 'Respiratory',  # 'Carbon monoxide poisoning',
         59: 'Respiratory',  # 'Respiratory - type unclassified (inc smoke inhalation)',
         60: 'Malignancy',  # 'Cancer (other than brain tumour)',
         70: 'Infection',  # 'Meningitis',
         71: 'Infection',  # 'Septicaemia',
         72: 'Infection',  # 'Infections - type unclassified',
         73: 'Acute blood loss/hypovolaemia',
         74: 'Liver failure (not self-poisoning)',
         75: 'Renal failure',
         76: 'Multi-organ failure',
         78: 'Burns',
         80: 'Known or suspected suicide',  # 'Alcohol poisoning',
         81: 'Known or suspected suicide',  # 'Paracetamol overdose',
         82: 'Known or suspected suicide',  # 'Other drug overdose, please specify',
         85: 'Known or suspected suicide',  # 'Self-poisoning - type unclassified'
     }
     ),
    ({'reason1', 'reason2'},
     {
         0: 'Not applicable',
         1: 'Logistical reason',  # 'Family permission not sought',
         2: 'Logistical reason',  # 'Family permission refused',
         3: 'Logistical reason',  # 'Permission refused by coroner',
         4: 'Logistical reason',  # 'Permission refused other',
         5: 'Organ not offered (eg. euro/living donor)',
         6: 'Organ inappropriate',  # 'Organ not present',
         7: 'Logistical reason',  # 'Offer withdrawn',
         9: 'Logistical reason',  # 'No permission',
         10: 'Donor unsuitable',  # 'Donor unsuitable – CoD',
         11: 'Donor unsuitable',  # 'Donor unsuitable – age',
         12: 'Donor unsuitable',  # 'Donor unsuitable – past history',
         13: 'Donor unsuitable',  # 'Donor recovered',
         14: 'Donor unsuitable',  # 'Non heart beating donor',
         15: 'Donor unsuitable',  # 'Brain stem tests not satisfied',
         16: 'Donor unsuitable',  # 'Donor unstable',
         17: 'Donor unsuitable',  # 'Donor unsuitable – size',
         18: 'Donor unsuitable',  # 'Donor arrested',
         19: 'Donor unsuitable',  # 'Donor unsuitable – other/unknown',
         20: 'Logistical reason',  # 'No suitable recipients',
         21: 'Logistical reason',  # 'No beds/staff/theatre',
         22: 'Logistical reason',  # 'No time',
         23: 'Logistical reason',  # 'Centre barred',
         24: 'Logistical reason',  # 'Centre already retrieving/transplanting',
         25: 'Logistical reason',  # 'Centre closed',
         26: 'Logistical reason',  # 'Centre criteria not achieved',
         27: 'Logistical reason',  # 'No blood for virology',
         28: 'Poor function',  # 'Poor function',
         29: 'Logistical reason',  # 'Other administrative reason',
         30: 'Organ inappropriate',  # 'Infection',
         31: 'Organ inappropriate',  # 'Contamination/damage in removal',
         32: 'Poor function',  # 'Poor function/ischaemic time',
         33: 'Recipient unfit/not appropriate',  # 'Clinical',
         34: 'Organ inappropriate',  # 'Tumour',
         35: 'Organ inappropriate',  # 'Anatomical',
         36: 'Poor function',  # 'Poor perfusion',
         37: 'Organ inappropriate',  # 'On perfusion machine',
         38: 'Organ inappropriate',  # 'Medication',
         39: 'Organ inappropriate',  # 'Other disease',
         40: 'Matching issue',  # 'HLA/ABO type',
         41: 'Matching issue',  # 'X-match positive',
         42: 'Matching issue',  # 'Unable to X-match',
         43: 'Matching issue',  # 'Better match required',
         44: 'Organ inappropriate',  # 'Organ damaged',
         45: 'Organ inappropriate',  # 'Contamination',
         46: 'Poor function',  # 'Ischaemia time too long – warm',
         47: 'Poor function',  # 'Ischaemia time too long – cold',
         48: 'Matching issue',  # 'Unable to X-match – no donor material',
         49: 'Matching issue',  # 'Unable to X-match – no recipient material',
         50: 'Recipient unfit/not appropriate',  # 'Recipient unfit',
         51: 'Recipient unfit/not appropriate',  # 'Recipient died',
         52: 'Recipient unfit/not appropriate',  # 'Recipient unavailable',
         53: 'Recipient unfit/not appropriate',  # 'Recipient refused',
         54: 'Recipient unfit/not appropriate',  # 'Recipient did not need tx',
         55: 'Logistical reason',  # 'Limited theatre time',
         56: 'Logistical reason',  # 'Recipient due to receive live donor tx',
         59: 'Logistical reason',  # 'Offered to national pool as payback',
         60: 'Logistical reason',  # 'Currently in tissue bank',
         61: 'Organ inappropriate',  # 'Infection in storage medium',
         62: 'Organ inappropriate',  # 'Expired in tissue bank',
         63: 'Epikeratophakia',
         64: 'Organ inappropriate',  # 'Tissue bank classify as unsuitable',
         65: 'Logistical reason',  # 'Issued from tissue bank for unknown recipient',
         66: 'Taken for hepatocytes',
         70: 'Logistical reason',  # 'Only taken for research use',
         71: 'Logistical reason',  # 'Poor weather',
         72: 'Logistical reason',  # 'Packaging',
         73: 'Logistical reason',  # 'Organ used elsewhere',
         74: 'Distance (euro)',
         75: 'Logistical reason',  # 'Offer waived',
         76: 'Logistical reason',  # 'No beds',
         77: 'Logistical reason',  # 'No staff',
         78: 'Logistical reason',  # 'No theatre',
         79: 'Logistical reason',  # 'Transport difficulties',
         81: 'Logistical reason',  # 'No response to fast track offer',
         82: 'Offer from europe for super-urgents',
         83: 'Heart retrived for valves only',
         84: 'Organ inappropriate',  # 'Used for research after declined by centres',
         85: 'Organ inappropriate',  # 'Fatty organ',
         86: 'Donor unsuitable',  # 'Donor unsuitable – virology',
         87: 'Donor unsuitable',  # 'Donor unsuitable – medical reason',
         90: 'Organ inappropriate',  # 'Organ unsuitable for tx',
         91: 'Unable to purify pancreas islets',
         92: 'Insufficient pancreas islets',
         93: 'Organ inappropriate',  # 'Whole organ cut down for tx',
         95: 'Logistical reason',  # 'Donor centre',
         96: 'Logistical reason',  # 'Recipient centre',
         97: 'Zone team felt organ not viable',
         101: 'Organ inappropriate',  # 'Organ too small',
         102: 'Organ inappropriate',  # 'Organ fibrotic',
         103: 'Insufficient distension with collagenase',
         104: 'Insufficient islet yield',
         105: 'Insufficient islet viability',
         106: 'Insufficient islet purity',
         107: 'Organ inappropriate',  # 'Packed cell volume too large',
         108: 'Organ inappropriate',  # 'Organ fatty infiltration',
         109: 'No kidney available'

     }
     ),
    ({'prd', 'rregprd'},
     {
         200: 'Other',  # 'Chronic renal failure, aetiology uncertain',
         210: 'Glomerulonephritis',  # 'Glomerulonephritis, histologically not examined',
         211: 'Glomerulonephritis',  # 'Severe nephrotic syndrome with focal sclerosis',
         212: 'Nephropathy',  # 'IgA nephropathy',
         213: 'Glomerulonephritis',  # 'Dense deposit disease',
         214: 'Nephropathy',  # 'Membranous nephropathy',
         215: 'Glomerulonephritis',  # 'Membrano – proliferative glomerulonephritis',
         216: 'Glomerulonephritis',  # 'Rapidly progressive GN without systemic disease',
         217: 'Glomerulonephritis',  # 'Focal segmental glomerulosclerosis with nephrotic syndrome in adults',
         219: 'Glomerulonephritis',  # 'Glomerulonephritis, histologically examined',
         220: 'Pyelonephritis/Interstitial nephritis',  # 'Pyelonephritis/Interstitial nephritis – cause not specified',
         221: 'Pyelonephritis/Interstitial nephritis',
         # 'Pyelonephritis/Interstitial nephritis associated with neurogenic bladder',
         222: 'Pyelonephritis/Interstitial nephritis',
         # 'Pyelonephritis/Interstitial nephritis due to con obs uropathy with/without V-U reflux',
         223: 'Pyelonephritis/Interstitial nephritis',
         # 'Pyelonephritis/Interstitial nephritis due to acquired obstructive uropathy',
         224: 'Pyelonephritis/Interstitial nephritis',
         # 'Pyelonephritis/Interstitial nephritis due to V-U reflux without obstruction',
         225: 'Pyelonephritis/Interstitial nephritis',  # 'Pyelonephritis/Interstitial nephritis due to urolithiasis',
         229: 'Pyelonephritis/Interstitial nephritis',  # 'Pyelonephritis/Interstitial nephritis due to other cause',
         230: 'Pyelonephritis/Interstitial nephritis',  # 'Tubulo Interstitial Nephritis (Not Pyelonephritis)',
         231: 'Nephropathy',  # 'Nephropathy due to analgesic drugs',
         232: 'Nephropathy',  # 'Nephropathy due to cisplatinum',
         233: 'Nephropathy',  # 'Nephropathy due to cyclosporin A',
         234: 'Nephropathy',  # 'Lead induced nephropathy (interstitial)',
         239: 'Nephropathy',  # 'Nephropathy caused by other specific drug',
         240: 'Cistic kidney disease',  # 'Cystic kidney disease – type unspecified',
         241: 'Cistic kidney disease',  # 'Polycystic kidneys, adult type (dominant type)',
         242: 'Cistic kidney disease',  # 'Polycystic kidneys, infantile (recessive type)',
         243: 'Cistic kidney disease',  # 'Medullary cystic disease, including nephronophthisis',
         249: 'Cistic kidney disease',  # 'Cystic kidney disease – other specified type',
         250: 'Nephropathy',  # 'Hereditary/Familial nephropathy – type unspecified',
         251: 'Glomerulonephritis',  # 'Hereditary nephritis with nerve deafness (Alports syndrome)',
         252: 'Other',  # 'Cystinosis',
         253: 'Other',  # 'Primary oxalosis',
         254: 'Other',  # 'Fabrys disease',
         259: 'Nephropathy',  # 'Hereditary nephropathy – other',
         260: 'Other',  # 'Congenital renal hypoplasia – type unspecified',
         261: 'Other',  # 'Oligomeganephronic hypoplasia',
         262: 'Other',  # 'Segmental renal hypoplasia',
         263: 'Other',  # 'Congenital renal dysplasia with or without urinary tract malformation',
         266: 'Other',  # 'Syndrome of agenesis of abdominal muscles',
         270: 'Renal vascular disease',  # 'Renal vascular disease – type unspecified ',
         271: 'Renal vascular disease',  # 'Renal vascular disease – malignant hypertension',
         272: 'Renal vascular disease',  # 'Renal vascular disease – hypertension',
         273: 'Renal vascular disease',  # 'Renal vascular disease – polyarteritis',
         274: 'Renal vascular disease',  # 'Wegeners granulomatosis',
         275: 'Renal vascular disease',  # 'Ischaemic renal disease/ cholesterol embolism',
         276: 'Glomerulonephritis',  # 'Glomerulonephritis related to liver cirrhosis',
         278: 'Glomerulonephritis',  # 'Cryoglobulinemic glomerulonephritis',
         279: 'Renal vascular disease',  # 'Renal vascular disease – classified',
         280: 'Diabetes',  # 'Diabetes – insulin dependent (Type I)',
         281: 'Diabetes',  # 'Diabetes – non-insulin dependent (Type II)',
         282: 'Other',  # 'Myelomatosis/Light chain deposit disease',
         283: 'Other',  # 'Amyloid',
         284: 'Glomerulonephritis',  # 'Lupus erythematosus',
         285: 'Glomerulonephritis',  # 'Henoch-Schonlein purpura',
         286: 'Glomerulonephritis',  # 'Goodpastures Syndrome',
         287: 'Other',  # 'Systemic sclerosis (Scleroderma)',
         288: 'Glomerulonephritis',  # 'Haemolytic Uraemic Syndrome (inc Moschowitz Syndrome)',
         289: 'Other',  # 'Multi-system disease – other',
         290: 'Other',  # 'Cortical or tubular necrosis',
         291: 'Other',  # 'Tuberculosis',
         292: 'Other',  # 'Gout',
         293: 'Nephropathy',  # 'Nephrocalcinosis & hypercalcaemic nephropathy',
         294: 'Nephropathy',  # 'Balkan nephropathy',
         295: 'Other',  # 'Kidney tumour',
         296: 'Other',  # 'Traumatic or surgical loss of kidney',
         298: 'Other',  # 'Other identified renal disorders'
     }
     )
]

BINARY_KEYS = {
    'Yes': 0,
    'No': 1,
    'Positive': 0,
    'Negative': 1,
    'Censored': 0,
    'Graft failure or death': 1,
    'Graft failure': 1,
    'Death of recipient': 1
}

def is_biolevel(column):
    """
    Check wether a column name refers to a biological level or not.
    """
    return bool(re.fullmatch("(alt|ast|amylase|creatinine|degfr)(_.*|$)", column))

def get_biolevel_columns(biolevel, df, temporal_columns_only=False):
    """
    Provides all the original columns corresponding to a given biological level (AST, creatinine, etc.).
    """
    idx     = [11, 12, 31, 32, 61, 62, 63, 71, 72, 73, 74, 75, 81, 82, 83, 84, 85]
    columns = [biolevel + '_' + str(i) for i in idx]
    if temporal_columns_only:
        columns = columns[:7]
    else:
        if biolevel == "creatinine":
            additional_column = ['dret_creat']
        elif biolevel == "degfr":
            additional_column = ['degfr']
        else:
            additional_column = list()
        columns = additional_column + columns
    return intersect_columns(columns, df)

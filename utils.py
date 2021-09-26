import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from hcuppy.elixhauser import ElixhauserEngine
from hcuppy.cci import CCIEngine
          
def add_ASL_codes(visits, ICD_col, ICPC_col, ASL_ICD_csv, ASL_ICPC_csv, ASL_names_csv):
    df = visits.copy()
    N = len(df)
    df['row_ID'] = np.arange(N)
    df['ICD_to_create_ASL'] = df[ICD_col].str.split('.').str[0]
    
    ASL_ICD_df = pd.read_csv(ASL_ICD_csv, sep=';').rename(columns={'Asl': 'ASL_ICD'}).drop(columns=['icd3'])
    ASL_ICPC_df = pd.read_csv(ASL_ICPC_csv, sep=';').rename(columns={'Asl': 'ASL_ICPC'})
    ASL_ICD_df['Sp'] = ASL_ICD_df['Sp'].replace('\r', np.nan).astype(float)

    for ASL_df, left_on, right_on in [(ASL_ICD_df, 'ICD_to_create_ASL', 'Icd'), (ASL_ICPC_df, ICPC_col, 'Icpc')]:
        df = df.merge(ASL_df, how='left', left_on=left_on, right_on=right_on)
        age_match = (np.floor(df['Potilaan ikä']) >= df['Alaika']) & (np.floor(df['Potilaan ikä']) <= df['Ylaika'])
        sex_match = (df['Sp'].isnull()) | (df['Sp'] == df['Potilaan sukupuoli'])
        df = df[df[right_on].isnull() | (age_match & sex_match)]
        df = df.drop(columns=['Alaika', 'Ylaika', 'Sp', right_on])
    
    assert len(df) == N
    assert set(np.arange(N)) == set(df['row_ID'])
    df = df.drop(columns=['ICD_to_create_ASL', 'row_ID'])
    df['ASL code'] = df['ASL_ICD'].fillna(df['ASL_ICPC']) # ICD tends to be more precise than ICPC
    
    # Adding ASL names/descriptions in Finnish and English
    ASL_names = pd.read_csv(ASL_names_csv, sep=';', encoding='latin-1').set_index('Asl')
    df['ASL nimi'] = df['ASL code'].replace(ASL_names['nimi'].to_dict())
    df['ASL'] = df['ASL code'].replace(ASL_names['name'].to_dict())
    
    return df

def read_unit_cost(filename):
    unit_cost = pd.read_excel(filename)
    unit_cost = unit_cost.melt(id_vars=['Toimintayksikkö', 'Ammatti'], var_name='Käyntityyppi', value_name='')
    unit_cost.columns = ['_'.join(['Kustannus', x]) for x in unit_cost.columns]
    unit_cost.rename(columns={'Kustannus_': 'Cost'}, inplace=True)
    return unit_cost

def add_cost_to_visits(visits, cost_filename):
    unit_cost = read_unit_cost(cost_filename)
    
    # Toimintayksikkö
    visits['Kustannus_Toimintayksikkö'] = 'Vastaanotto'
    toimintayksiköt = {'Päivystys': ['PÄIVYSTYSKÄYNTI PERUSTYÖAIKANA LÄÄK.VASTAANOTOLLA', 'Päiväpoliklinikan lääkärikäynti', 'ARKIPÄIVYSTYSKÄYNTI LÄÄKÄRIN VASTAANOTOLLA', 'VIIKONLOPUN JA ARKIYÖN P-KÄYNTI LÄÄK VO:LLA'],
                       'Äitiysneuvola': ['KÄYNTI NEUVOLALÄÄKÄRIN VASTAANOTOLLA', 'KÄYNTI E-NLAN LÄÄKÄRIN VASTAANOTOLLA'],
                      'Koulu- ja opiskelijaterv.': ['KÄYNTI KOULU/OPISKELUTERV.HUOLLON LÄÄKÄRIN VO:LLA']}
    for k, v in toimintayksiköt.items():
        visits['Kustannus_Toimintayksikkö'] = visits['Kustannus_Toimintayksikkö'].mask(visits['Käyntiluokan nimi'].isin(v), k)
        
    # Ammatti
    visits['Kustannus_Ammatti'] = visits['Käynnin vastaanottajan ammattiryhmän nimi'].str.split(',').str[0].str[:-1]
    visits['Kustannus_Ammatti'] = visits['Kustannus_Ammatti'].mask(visits['Kustannus_Ammatti'].str.contains('Lääkäri', case=False, regex=False, na=False), 'Lääkäri')
    visits.loc[~visits['Kustannus_Ammatti'].isin(unit_cost['Kustannus_Ammatti'].unique()), 'Kustannus_Ammatti'] = 'Muu ammattiryhmä'

    # Käyntityyppi
    visits['Kustannus_Käyntityyppi'] = visits['Käynnin yhteystavan nimi'].map(
                        {'Käynti vastaanotolla': 'Vastaanottokäynti', 'Puhelinyhteys': 'Puhelu', 'Sähköinen yhteys': 'Kirje tai sähk. Yhteydenotto',
                            'Asiakirjamerkintä ilman asiakaskontaktia': 'Toimistotyö ilman kontaktia', 'Kirje': 'Kirje tai sähk. Yhteydenotto', 'Konsultaatio': 'Konsultaatio',
                        'Kotikäynti': 'Kotikäynti'})\
                        .fillna('Toimistotyö ilman kontaktia') # Muu / Sairaalakäynti
    käyntityypit = {'Vastaanottokäynti': ['KÄYNTI SAIRAANHOITAJAN VASTAANOTOLLA', 'PERUSTYÖAJAN KÄYNTI LÄÄK.VAST.OTOLLA(AJANVARAUS)', 'PÄIVYSTYSKÄYNTI PERUSTYÖAIKANA LÄÄK.VASTAANOTOLLA', 'Päiväpoliklinikan lääkärikäynti', 'ARKIPÄIVYSTYSKÄYNTI LÄÄKÄRIN VASTAANOTOLLA', 'VIIKONLOPUN JA ARKIYÖN P-KÄYNTI LÄÄK VO:LLA', 'KÄYNTI NEUVOLALÄÄKÄRIN VASTAANOTOLLA', 'KÄYNTI E-NLAN LÄÄKÄRIN VASTAANOTOLLA', 
                                         'KÄYNTI KOULU/OPISKELUTERV.HUOLLON LÄÄKÄRIN VO:LLA', 'KÄYNTI TERVEYDENHOITAJAN VASTAANOTOLLA'],
                       'Puhelu': ['PUHELINKONTAKTI PERUSTYÖAIKANA'], 'Ryhmätoiminta': ['RYHMÄNEUVONTA'],
                   'Konsultaatio': ['PAPERIKONSULTAATIO', 'KONSULTAATIO'], 'Toimistotyö ilman kontaktia': ['Lääkärin merkintä']}
    for k, v in käyntityypit.items():
        visits['Kustannus_Käyntityyppi'] = visits['Kustannus_Käyntityyppi'].mask(visits['Käyntiluokan nimi'].isin(v), k)
        
    cost_cols = ['Kustannus_Toimintayksikkö', 'Kustannus_Ammatti', 'Kustannus_Käyntityyppi']
    visits = visits.merge(unit_cost, on=cost_cols, how='left').drop(columns=cost_cols)
    
    # Price index correction +10.1%
    visits['Cost'] = visits['Cost'] * 1.101
    assert visits['Cost'].isnull().sum() == 0
    return visits
                
def read_visits_csv(folder, filenames, include_cost=False, cost_filename=None):
    dtype={'Toimipisteen lyhenne': 'category', 'Käyntiluokan koodi': 'category', 'Vastaanottajan nimi': 'category', 'Potilaan lista käyntipäivänä': 'string', 'Käyntipäivä': 'string', 'Käyntiluokan nimi': 'category', 'Potilaan ikä': 'string',
           'Potilaan sukupuoli': 'boolean', 'Potilasnumero': 'Int64', 'Käyntiryhmän koodi': 'category', 'Käyntiryhmän nimi': 'category', 'Toimipisteen nimi': 'category', 'Käynnin ICD10-diagnoosi 1': 'category', 'Käynnin ICPC-diagnoosi 1': 'category',
           'Käynnin kesto minuutteina': 'Int64', 'Käynnin vastaanottajan ammattiryhmän nimi': 'category', 'Käynnin kiireellisyys nimi': 'category', 'Käynnin yhteystavan nimi': 'category', 'Käynnin luonteen nimi': 'category',
           'Käynnin palvelumuodon nimi': 'category', 'Käynnin viikonpäivä': 'category', 'Hoidon tarpeen arviointi': 'boolean', 'Kotikunta2': 'category', 'Ensikäynti': 'boolean', 'Käynnin kävijäryhmä nimi': 'category',
           'Hoidon tarpeen arviointiteksti': 'string'}

    visits = pd.read_csv(folder + filenames['visits'], sep=';', encoding='latin-1', error_bad_lines=True, dtype=dtype) # error_bad_lines=False, parse_dates=['Käyntipäivä']
    visits['Käyntipäivä'] = pd.to_datetime(visits['Käyntipäivä'], format='%d.%m.%Y')
    visits = visits[(visits['Käyntipäivä'] >= '2017-08-07') & (visits['Käyntipäivä'] <= '2018-10-03')]
    visits['Potilaan ikä'] = visits['Potilaan ikä'].str.replace(',', '.').astype(float)

    assert (visits['Vastaanottajan nimi.1'] == visits['Vastaanottajan nimi']).all()
    visits.drop(columns=['Vastaanottajan nimi.1'], inplace=True) # is duplicate of 'Vastaanottajan nimi'

    visits['Hoidon tarpeen arviointiteksti'] = visits['Hoidon tarpeen arviointiteksti'].str.replace('{', 'ä', regex=False).str.replace('[', 'Ä', regex=False)
    visits['Klinik contact time'] = pd.to_datetime(visits['Hoidon tarpeen arviointiteksti'].str.split('Lähetetty').str[-1], format=': %d.%m.%Y klo %H.%M', errors='coerce', infer_datetime_format=True)
    
    visits.drop_duplicates(inplace=True)
    visits = visits.sort_values(by=['Potilasnumero', 'Käyntipäivä']).reset_index(drop=True)
    
    if include_cost:
        visits = add_cost_to_visits(visits, cost_filename)
        
    ICD10 = pd.read_excel(folder + filenames['ICD']).rename(columns={'LongName': 'ICD_explanation', 'ALONG:EnsisijainenICPCkoodiMiehillä': 'ICD_to_ICPC_male', 'ALONG:EnsisijainenICPCkoodiNaisilla': 'ICD_to_ICPC_female'}).rename(columns={'ParentId': 'ICD_ParentId'})
    ICPC = pd.read_excel(folder + filenames['ICPC']).rename(columns={'LongName': 'ICPC_explanation', 'ALONG:Ensisijainen ICD-10': 'ICPC_to_ICD'})
    ICPC = ICPC[ICPC['HierarchyLevel'] == 1].merge(ICPC.loc[ICPC['HierarchyLevel'] == 0, ['ICPC_explanation', 'CodeId']].rename(columns={'ICPC_explanation': 'ICPC_parent_explanation', 'CodeId': 'ParentId'}), how='left', on='ParentId', validate='many_to_one').rename(columns={'ParentId': 'ICPC_ParentId'})
    visits = visits.merge(ICD10[['CodeId', 'ICD_to_ICPC_male', 'ICD_to_ICPC_female']], how='left', validate='many_to_one',
                                left_on='Käynnin ICD10-diagnoosi 1', right_on='CodeId').drop(columns=['CodeId'])
    visits = visits.merge(ICPC[['CodeId', 'ICPC_to_ICD']], how='left', validate='many_to_one',
                                left_on='Käynnin ICPC-diagnoosi 1', right_on='CodeId').drop(columns=['CodeId'])
    visits['ICD_to_ICPC'] = visits['ICD_to_ICPC_female'].where(cond=visits['Potilaan sukupuoli'], other=visits['ICD_to_ICPC_male'])
    visits.drop(columns={'ICD_to_ICPC_female', 'ICD_to_ICPC_male'}, inplace=True)
    
    visits['ICD'] = visits['Käynnin ICD10-diagnoosi 1'].fillna(visits['ICPC_to_ICD'])
    visits['ICPC'] = visits['Käynnin ICPC-diagnoosi 1'].fillna(visits['ICD_to_ICPC'])
    
    visits = visits.merge(ICD10[['CodeId', 'ICD_explanation', 'ICD_ParentId']], how='left', validate='many_to_one',
                                left_on='ICD', right_on='CodeId').drop(columns=['CodeId'])
    visits = visits.merge(ICPC[['CodeId', 'ICPC_ParentId', 'ICPC_explanation', 'ICPC_parent_explanation']], how='left', validate='many_to_one',
                                left_on='ICPC', right_on='CodeId').drop(columns=['CodeId'])
    
    visits = add_ASL_codes(visits, 'Käynnin ICD10-diagnoosi 1', 'Käynnin ICPC-diagnoosi 1', folder + filenames['ASL_ICD'], folder + filenames['ASL_ICPC'], folder + filenames['ASL_names'])
    
    visits['Klinik'] = ((visits['Käyntiluokan nimi'] == 'SÄHKÖINEN HTA') | visits['Hoidon tarpeen arviointiteksti'].str.contains('VAIVA TAI SAIRAUS', case=True, regex=False, na=False)) & visits['Hoidon tarpeen arviointi']
    visits['Phone'] = (visits['Käynnin yhteystavan nimi'] == 'Puhelinyhteys') & visits['Hoidon tarpeen arviointi']
    visits['Phone'] = visits['Phone'] & ~visits['Klinik'] # In some rare cases we have 'SÄHKÖINEN HTA' and 'Puhelinyhteys'. These seem to be Klinik based on 'Hoidon tarpeen arviointiteksti'
    visits['Walk-in'] = (visits['Käynnin yhteystavan nimi'] == 'Käynti vastaanotolla') & visits['Hoidon tarpeen arviointi'] 
    visits['Walk-in'] = visits['Walk-in'] & ~visits['Klinik'] # In some rare cases we have 'SÄHKÖINEN HTA' and 'Käynti vastaanotolla'. These seem to be Klinik based on 'Hoidon tarpeen arviointiteksti'
    visits['Index visit'] = np.nan
    visits['Index visit'] = visits['Index visit'].mask(visits['Klinik'], 'Klinik').mask(visits['Phone'], 'Phone').mask(visits['Walk-in'], 'Walk-in')
    
    visits.loc[visits['Klinik'] | visits['Phone'], 'Cost'] = 12 * 1.101 # Setting the contacting cost same as they should be compared separately if different
    
    visits = visits.sort_values(['Potilasnumero', 'Käyntipäivä', 'Hoidon tarpeen arviointi'], ascending=[True, True, False])
    return visits.rename(columns={'Potilaan ikä': 'Age', 'Potilaan sukupuoli': 'Sex', 'Potilasnumero': 'Patient ID', 'Käynnin kiireellisyys nimi': 'Kiireellisyys', 'Käynnin luonteen nimi': 'Luonne', 'Käyntipäivä': 'Time',
                                  'Hoidon tarpeen arviointi': 'Care needs assessment', 'Ensikäynti': 'First visit'})

def create_Klinik_visits_from_contacts(visits, contacts):
    contacts_to_add = contacts[set(visits).intersection(set(contacts))].copy()
    contacts_to_add = contacts_to_add[contacts_to_add['Klinik'] & contacts_to_add['Patient ID'].isin(visits['Patient ID'].unique())]
    for col, value in [('Cost', 12 * 1.101), ('Index visit', 'Klinik'), ('Phone', False), ('Walk-in', False),
                       ('Care needs assessment', True), ('Contact', True), ('Käyntiluokan nimi', 'SÄHKÖINEN HTA')]:
        contacts_to_add[col] = value
    contacts_to_add = visits[visits['Klinik']].append(contacts_to_add)
    contacts_to_add = contacts_to_add.drop_duplicates(subset=['Patient ID', 'Time'], keep=False)
    contacts_to_add = contacts_to_add[contacts_to_add['Contact'] == True]#.drop(columns=['Contact'])
    
    visits = visits.append(contacts_to_add).sort_values(['Patient ID', 'Time', 'Care needs assessment'], ascending=[True, True, False]).reset_index(drop=True)
    for col in ['Age', 'Sex']:
        visits[col] = visits[col].fillna(visits.groupby('Patient ID')[col].transform('mean'))
    visits['Contact'] = visits['Contact'].fillna(False)
    return visits

def read_contacts_csv(filename, include_cost=False, cost_filename=None):
    dtype={'Ensikäynti': 'boolean', 'Kiireellisyys': 'category', 'Luonne': 'category', 'Tulos': 'category', 'Toimipiste': 'category', 'Potilasnumero': 'Int64', 'Tila': 'category', 'Yhteydenotto-Tapahtuma': 'string',
       'Kello': 'string', 'Yhteydenotto-Päivä': 'string', 'Yo-av vastaanottaja': 'category', 'Yo-av käyntityyppi': 'category', 'Yo-av ammattiryhmän nimi': 'category', 'Yo-av päivä': 'object', 'Yo-av vastaanottaja.1': 'category',
       'Yhteydenoton lisätieto': 'string', 'Yo-av lkm': 'Int64'}
    contacts = pd.read_csv(filename, sep=';', dtype=dtype, encoding='latin-1')

    contacts['Yo-av päivä'] = pd.to_datetime(contacts['Yo-av päivä'], format='%d.%m.%Y')
    contacts['Yhteydenotto aika'] = contacts['Yhteydenotto-Päivä'] + ' ' + contacts['Kello']
    contacts['Yhteydenotto aika'] = pd.to_datetime(contacts['Yhteydenotto aika'], format='%Y-%m-%d %H:%M')
    contacts['Yhteydenotto-Päivä'] = pd.to_datetime(contacts['Yhteydenotto-Päivä'], format='%Y-%m-%d')
    
    contacts = contacts[(contacts['Yhteydenotto-Päivä'] >= '2017-08-07') & (contacts['Yhteydenotto-Päivä'] <= '2018-10-03')]

    assert ((contacts['Yo-av vastaanottaja.1'] == contacts['Yo-av vastaanottaja']) | (contacts['Yo-av vastaanottaja.1'].isnull() & contacts['Yo-av vastaanottaja'].isnull())).all()
    contacts.drop(columns=['Yo-av vastaanottaja.1'], inplace=True) # is duplicate of 'Yo-av vastaanottaja'

    contacts.drop_duplicates(inplace=True)
    contacts = contacts.sort_values(by=['Potilasnumero', 'Yhteydenotto aika']).reset_index(drop=True)
    
    if include_cost:
        # Cost of contact is zero since these events overlap with the other data
        contacts['Cost'] = phone_cost = 0.0
        assert contacts['Cost'].isnull().sum() == 0
        
    contacts['Klinik'] = (contacts['Yhteydenotto-Tapahtuma'].str.contains('ehta', case=False, na=False) | contacts['Yhteydenoton lisätieto'].str.contains('ehta', case=False, na=False)) & (contacts['Tila'] == '3')
    contacts['Phone'] = False # (~contacts['Klinik']) & (contacts['Tila'] == '3')
    return contacts.rename(columns={'Potilasnumero': 'Patient ID', 'Ensikäynti': 'First visit', 'Yhteydenotto-Päivä': 'Time'})

def read_bookings_csv(filename, include_cost=False, cost_filename=None):
    dtype = {'Ajanvaraus-Päivä': 'string', 'Ajanvaraus-Klo': 'string', 'Käyntityyppi': 'category', 'Vastaanottaja': 'category', 'Sähköinen-Asiointi': 'boolean', 'Toimipiste lyh': 'category', 'Toimipisteen nimi': 'category',
             'Ajanvaraus-Tallennusaika': 'string', 'Potilasnumero': 'Int64', 'Varattu aika': 'string'}
    bookings = pd.read_csv(filename, sep=';', dtype=dtype, encoding='latin-1')

    bookings['Varattu aika'] = bookings['Ajanvaraus-Päivä'] + ' ' + bookings['Ajanvaraus-Klo']
    for datetime_col in ['Varattu aika', 'Ajanvaraus-Tallennusaika']:
        bookings[datetime_col] = pd.to_datetime(bookings[datetime_col], format='%Y-%m-%d %H:%M:%S') 
    bookings['Ajanvaraus-Päivä'] = pd.to_datetime(bookings['Ajanvaraus-Päivä'], format='%Y-%m-%d')

    bookings.drop_duplicates(inplace=True)

    bookings['Ajanvaraus-Tallennuspäivä'] = bookings['Ajanvaraus-Tallennusaika'].dt.normalize()
    bookings['Waiting time'] = (bookings['Varattu aika'] - bookings['Ajanvaraus-Tallennusaika']).apply(lambda x: x.total_seconds())
    bookings['Waiting time'] = bookings['Waiting time'] / (60*60*24)
    
    if include_cost:
        # The booking itself has zero cost, the cost of booking and the booked appointment will be included in the contact/visit data
        bookings['Cost'] = 0.0
        
    return bookings.rename(columns={'Potilasnumero': 'Patient ID'})#.rename(columns={'Sähköinen-Asiointi': 'Klinik'})

def add_time_columns(df, from_col, new_cols=['Hour', 'Weekday']):
    if 'Hour' in new_cols: df['Hour'] = df[from_col].dt.hour
    if 'Weekday' in new_cols: df['Weekday'] = df[from_col].dt.day_name()
    return df

def show_value_counts(df, dtype='category', limits = [5, 20]):
    for col in df.select_dtypes(dtype).columns:
        n_unique = df[col].nunique()
        if n_unique < limits[0]:
            print(col, '...', dict(df[col].astype(str).value_counts(dropna=False)))
        elif n_unique < limits[1]:
            print(col, '...', list(df[col].unique()))
        else:
            print(col, '...', n_unique)
            
def add_age_groups(df, age_sex_interaction):
    if age_sex_interaction:
        dummies = pd.get_dummies(df['Sex'].replace({1: 'Female', 0: 'Male'}) + ' ' + pd.cut(df['Age'], bins=[0, 7, 18, 26, 40, 55, 65, 75, 85, 90, 200], right=False, include_lowest=True).astype(str), drop_first=True) # THL tarvevakionti sisältää myös 0-2 erikseen
    else:
        dummies = pd.get_dummies(pd.cut(df['Age'], bins=[0, 7, 18, 26, 40, 55, 65, 75, 85, 90, 200], right=False, include_lowest=True).astype(str), drop_first=True) # THL tarvevakionti sisältää myös 0-2 erikseen
    df = df.merge(dummies, left_index=True, right_index=True, validate='one_to_one')
    #df['Male_health_check'] = df['Age'].astype(int).isin([50, 55, 60, 65]) & ~df['Sex'].astype('boolean')
    #df['Female_health_check'] = df['Age'].astype(int).isin([30, 35, 40, 45, 50, 55, 60, 65]) & df['Sex'].astype('boolean')
    return df, dummies.columns.to_list()

def make_dummies(df, col, drop_first):
    dummies = pd.get_dummies(df[col], drop_first=drop_first)
    df = df.merge(dummies, left_index=True, right_index=True, validate='one_to_one')
    return df, dummies.columns.to_list()

def add_user(df, include_walk_in=False):
    df['Klinik_user'] = df.groupby('Patient ID')['Klinik'].transform('max')
    df['Phone_user'] = df.groupby('Patient ID')['Phone'].transform('max')
    if include_walk_in: df['Walk-in_user'] = df.groupby('Patient ID')['Walk-in'].transform('max')
    df['User'] = 'Other'
    if include_walk_in: df.loc[df['Walk-in_user'] == 1, 'User'] = 'Walk-in'
    df.loc[df['Phone_user'] == 1, 'User'] = 'Phone'
    df.loc[df['Klinik_user'] == 1, 'User'] = 'Klinik'
    df['User'] = df['User'].astype('category')
    return df

def update_from_bookings(visits, bookings, ASL_or_episode, time_col):
    bookings_copy = bookings.copy()
    by_day = visits.groupby(['Patient ID', time_col])[ASL_or_episode].first().dropna()
    bookings_copy = bookings_copy.merge(by_day.rename(f'Booked {ASL_or_episode}').reset_index(), how='left', left_on=['Patient ID', 'Ajanvaraus-Päivä'], right_on=['Patient ID', time_col], validate='many_to_one').drop(columns=[time_col])
    bookings_copy = bookings_copy.merge(by_day.rename(f'Booking {ASL_or_episode}').reset_index(), how='left', left_on=['Patient ID', 'Ajanvaraus-Tallennuspäivä'], right_on=['Patient ID', time_col], validate='many_to_one').drop(columns=[time_col])
    for booking_time_col, ASL_col in [('Ajanvaraus-Tallennuspäivä', f'Booked {ASL_or_episode}'), ('Ajanvaraus-Päivä', f'Booking {ASL_or_episode}')]:
        temp = bookings_copy.dropna(subset=[ASL_col]).rename(columns={booking_time_col: time_col}).drop_duplicates(subset=['Patient ID', time_col])
        visits = visits.merge(temp[['Patient ID', time_col, ASL_col]], how='left', on=['Patient ID', time_col], validate='many_to_one')
    # If there is an ASL code from a booked appointment regarding a care needs assesment, the ASL code of the care needs assesment is replaced with the that of the booked appointment (more precise info/diagnosis on the patient later on)
    if ASL_or_episode == 'ASL':
        visits['ASL'] = visits['ASL'].mask((visits['Klinik'] | visits['Phone'] | visits['Walk-in']) & visits['Booked ASL'].notnull(), visits['Booked ASL'])
    # If the ASL/episode code is missing, we will use 1. Booking code 2. Booked code
    visits[ASL_or_episode] = visits[ASL_or_episode].mask(visits[ASL_or_episode].isnull() & visits[f'Booking {ASL_or_episode}'].notnull(), visits[f'Booking {ASL_or_episode}'])
    visits[ASL_or_episode] = visits[ASL_or_episode].mask(visits[ASL_or_episode].isnull() & visits[f'Booked {ASL_or_episode}'].notnull(), visits[f'Booked {ASL_or_episode}'])
    return visits

def code_fill_between(visits, code_col):
    ffill = visits.groupby('Patient ID', sort=False)[code_col].transform('ffill')
    bfill = visits.groupby('Patient ID', sort=False)[code_col].transform('bfill')
    visits[code_col] = visits[code_col].mask(ffill == bfill, ffill)
    return visits

def code_same_day(visits, code_col, time_col):
    ASL_by_day = visits.groupby(['Patient ID', time_col])[code_col].transform('first')
    visits[code_col] = visits[code_col].fillna(ASL_by_day)
    return visits

def create_episode_numbers(visits, time_col):
    # Creating column for each ASL code, where 0 indicates the existance of that code and np.nan otherwise
    ASL_dummies = pd.get_dummies(visits['ASL'], drop_first=False)
    ASL_codes = ASL_dummies.columns.to_list()
    ASL_dummies = ASL_dummies.where(ASL_dummies == 1) - 1
    visits = visits.merge(ASL_dummies, left_index=True, right_index=True, validate='one_to_one')
    # replacing the 0 with 1 if it is also a Klinik/Phone visit
    for col in ASL_codes:
        visits.loc[(visits['Klinik'] | visits['Phone'] | visits['Walk-in']) & visits[col].notnull(), col] = 1.
    # However, if two care needs assesments are less than 14 days from each other, the second episode will be merged with the first
    multiple_assesments = visits[['Patient ID', time_col] + ASL_codes].copy()
    multiple_assesments = multiple_assesments.fillna(0.).groupby('Patient ID').rolling(window='14D', on=time_col, closed='both').sum()
    multiple_assesments = (multiple_assesments.reset_index(level=0, drop=True)[ASL_codes] > 1) & (visits[ASL_codes] == 1)
    visits[ASL_codes] = visits[ASL_codes].mask(multiple_assesments, 0.)
    # This allows for a cumulative sum, so that each Klinik/Phone visit and the corresponding ASL visits after it get the same number
    visits[ASL_codes] = visits.groupby('Patient ID')[ASL_codes].transform('cumsum')
    visits[ASL_codes] = visits[ASL_codes].replace(0, np.nan)
    # If a visit falls between two visits with same ASL episode, it is included in this episode
    ffill = visits.groupby('Patient ID', sort=False)[ASL_codes].transform('ffill')
    bfill = visits.groupby('Patient ID', sort=False)[ASL_codes].transform('bfill')
    ASL_null = visits['ASL'].isnull()
    between_ffill_bfill = np.logical_and(np.array(ffill == bfill), np.array(ASL_null).reshape(-1,1)) #(ffill == bfill) & others_were_null
    visits[ASL_codes] = visits[ASL_codes].mask(between_ffill_bfill, ffill)
    # If a visit is during the same day as another visit with an ASL episode, it is included in this episode
    same_day_ASL = visits.groupby(['Patient ID', time_col], sort=False)[ASL_codes].transform('max')
    same_day_ASL_mask = np.logical_and(np.array(same_day_ASL.notnull()), np.array(ASL_null).reshape(-1,1))
    visits[ASL_codes] = visits[ASL_codes].mask(same_day_ASL_mask, same_day_ASL)
    # Episodes are named "ASLCODE-number", which is unique identifier when used together with "Patient ID"
    visits['Episode'] = visits[ASL_codes].idxmax(axis=1).fillna('NO_ASL') + '-' + visits[ASL_codes].max(axis=1).fillna(0).astype(int).astype(str)
    visits['Episode'] = visits['Episode'].replace('NO_ASL-0', np.nan)
    return visits.drop(columns=ASL_dummies)

def create_episode_tails(visits, time_col, urgency_col, differing_tail_lengths):
    visits['Urgency'] = visits[urgency_col].mask(~(visits['Klinik'] | visits['Phone'] | visits['Walk-in']))
    episode_start_ends = visits.groupby(['Patient ID', 'Episode'])[[time_col, 'Urgency']].agg({time_col: ['min', 'max'], 'Urgency': 'first'})
    
    episode_tail = '60D'
    episode_tails = [('Päivystys', '14D'), ('1-7 päivää', '30D'), ('Ei-kiireellinen hoidon tarve', '60D')]
    episode_max_duration = '182D'
    
    if differing_tail_lengths:
        for urgency, tail_length in episode_tails:
            #print(visits['Episode'].notnull().sum())
            episode_ends = episode_start_ends[episode_start_ends[('Urgency', 'first')] == urgency][time_col].drop(columns=['min']).reset_index().rename(columns={'Episode': 'Episode tail'}).sort_values('max')
            episode_ends['Patient ID'] = episode_ends['Patient ID'].astype('Int64')
            visits = pd.merge_asof(visits.sort_values(time_col), episode_ends, by='Patient ID', left_on=time_col, right_on='max', direction='backward', tolerance=pd.Timedelta(tail_length))
            tail_fill = (visits['ASL'].isnull() | (visits['ASL'] == visits['Episode tail'].str.split('-').str[:-1].str.join('-'))) & visits['Episode'].isnull() & visits['Index visit'].isnull()
            visits['Episode'] = visits['Episode'].mask(tail_fill, visits['Episode tail'])
            visits = visits.drop(columns=['Episode tail', 'max'])
    else:
        #print(visits['Episode'].notnull().sum())
        episode_ends = episode_start_ends[time_col].drop(columns=['min']).reset_index().rename(columns={'Episode': 'Episode tail'}).sort_values('max')
        episode_ends['Patient ID'] = episode_ends['Patient ID'].astype('Int64')
        visits = pd.merge_asof(visits.sort_values(time_col), episode_ends, by='Patient ID', left_on=time_col, right_on='max', direction='backward', tolerance=pd.Timedelta(episode_tail))
        tail_fill = (visits['ASL'].isnull() | (visits['ASL'] == visits['Episode tail'].str.split('-').str[:-1].str.join('-'))) & visits['Episode'].isnull() & visits['Index visit'].isnull()
        visits.loc[tail_fill, 'Episode'] = visits.loc[tail_fill, 'Episode tail']
        visits = visits.drop(columns=['Episode tail', 'max'])

    #print(visits['Episode'].notnull().sum())
    episode_start_ends = episode_start_ends[time_col].reset_index()
    episode_start_ends['Patient ID'] = episode_start_ends['Patient ID'].astype('Int64')
    visits = visits.merge(episode_start_ends.drop(columns=['max']), how='left', on=['Patient ID', 'Episode'])
    visits.loc[(visits[time_col] - visits['min']) > episode_max_duration, 'Episode'] = np.nan
    return visits

def create_ASL_episodes(visits, bookings=[], time_col='Time', urgency_col='Kiireellisyys', HTA_col='Care needs assessment', differing_tail_lengths=False):
    # The ICPC code A98 is used quite a lot (low information) which can prevent it from being part of an episode, therefore it is removed
    visits['orig_ASL'] = visits['ASL']
    visits['ASL'] = visits['ASL'].mask((visits['Käynnin ICPC-diagnoosi 1'] == 'A98') & visits['ASL'].isin(['Healthy adult', 'Healthy child or teen']))
    
    visits = visits.sort_values(['Patient ID', time_col, HTA_col], ascending=[True, True, False])
    
    if len(bookings) > 0:
        visits = update_from_bookings(visits, bookings, 'ASL', time_col)

    # If a visit falls between two visits with same ASL code, it gets that ASL code
    visits = code_fill_between(visits, 'ASL')
    # If a visit has no ASL code, it gets the ASL code from another visit during the same day (if there is any)
    visits = code_same_day(visits, 'ASL', time_col)
    
    # Creating the episode numbering and names (per ASL code and patient)
    # Episodes are named "ASLCODE-number", which is unique identifier when used together with "Patient ID"
    visits = create_episode_numbers(visits, time_col)
    
    if len(bookings) > 0:
        visits = update_from_bookings(visits, bookings, 'Episode', time_col)
        
    # If a visit falls between two visits with the same episode code, it belogns to the same episode
    visits = code_fill_between(visits, 'Episode')
    # If a visit has no episode code, it gets the episode code from another visit during the same day (if there is any)
    visits = code_same_day(visits, 'Episode', time_col)
    
    # Adding some logic based on the length of the episode.
    # This includes a maximum episode length and a length for the tail of episode depending on the urgency of the index visit (tail includes the following visits within certain time after the last identified ASL episode visit)
    visits = create_episode_tails(visits, time_col, urgency_col, differing_tail_lengths)
    
    visits['Episode_ASL'] = visits['Episode'].str.split('-').str[:-1].str.join('-')
    visits = visits.drop(columns=['min']).sort_values(['Patient ID', time_col, HTA_col], ascending=[True, True, False])
    
    return visits

def add_rolling_utilisation_sum(visits, window='7D'):
    temp = visits.groupby('Time')['Index visit'].value_counts().rename('Count').reset_index()
    temp = temp.pivot(columns='Index visit', values='Count', index='Time').sort_index().rolling(window).sum().reset_index().drop_duplicates()
    temp[f'Klinik+Phone {window}'] = temp['Klinik'] + temp['Phone']
    visits = visits.merge(temp[['Time', f'Klinik+Phone {window}']], how='left', on=['Time'], validate='many_to_one')
    return visits

def add_elixhauser(df, include_dummies=False):
    ee = ElixhauserEngine()
    df['Elixhauser'] = df.groupby('Patient ID')['ICD'].transform(lambda x: ee.get_elixhauser(list(x.dropna()))['mrtlt_scr'] if (len(x.dropna()) > 0) else np.nan)
    df['Elixhauser nan'] = df['Elixhauser'].isnull()
    df['Elixhauser'] = df['Elixhauser'].fillna(0)
    df['Elixhauser+'] = df['Elixhauser'].mask(df['Elixhauser'] < 0).fillna(.0)
    df['Elixhauser-'] = df['Elixhauser'].mask(df['Elixhauser'] > 0).fillna(.0)
    dummy_cols = []
    if include_dummies:
        df['Elixhauser comorbs'] = df.groupby('Patient ID')['ICD'].transform(lambda x: str(ee.get_elixhauser(list(x.dropna()))['cmrbdt_lst']))
        df['Elixhauser comorbs'] = df['Elixhauser comorbs'].str.replace(r"[\[,'\]]", '', regex=True).str.split()
        mlb = MultiLabelBinarizer()
        elixhauser_dummies = pd.DataFrame(mlb.fit_transform(df['Elixhauser comorbs']), columns=mlb.classes_, index=df.index)
        dummy_cols = dummy_cols + list(elixhauser_dummies.columns)
        df = df.merge(elixhauser_dummies, left_index=True, right_index=True, validate='one_to_one')
        df = df.drop(columns=['Elixhauser comorbs'])
    return df.drop(columns=['Elixhauser']), dummy_cols

def add_chronic(df, groupby_cols, new_col='Chronic', agg=np.max):
    ce = CCIEngine()
    df[new_col] = df.groupby(groupby_cols)['ICD'].transform(lambda x: agg([int(x['is_chronic']) for x in ce.get_cci(list(x.dropna()))]) if (len(x.dropna()) > 0) else 0)
    return df

def create_ICPC_dummies(df, ICPC_N, include_A98=True):
    if ICPC_N > 0:
        ICPC_N_codes = df['ICPC'].value_counts()[:ICPC_N].index.to_list()
        df.loc[~df['ICPC'].isin(ICPC_N_codes) & df['ICPC'].notnull(), 'ICPC'] = 'OTHER'
        if not include_A98: df.loc[df['ICPC'] == 'A98', 'ICPC'] = pd.NA
        df, dummy_cols = make_dummies(df, 'ICPC', drop_first=False)
    elif ICPC_N == -1:
        df, dummy_cols = make_dummies(df, 'ICPC_parent_explanation', drop_first=False)
    else:
        ICPC_N = -ICPC_N
        ICPC_N_codes = df['ICPC'].value_counts()[:ICPC_N].index.to_list()
        df['ICPC'] = df['ICPC'].where(df['ICPC'].isin(ICPC_N_codes), df['ICPC_parent_explanation'])
        df, dummy_cols = make_dummies(df, 'ICPC', drop_first=False)
    return df, dummy_cols
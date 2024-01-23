
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import norm
from statistics import pstdev, mean
import re

df = ''

def init_db(path):
    df = pd.read_excel(path, na_values='-')


print(df.head())
print(df.columns)

tags = {}
codari = {}

def deal_data(df, index, cname):
    if len(df) > 31:
        df = df.drop(index=index)
        df.reset_index(drop=True, inplace=True)
        return df
    else:
        df.loc[index, cname] = df[cname].mean()
        return df

def no_data(df, cname):
    indices = df[df[cname].isna()].index
    return indices

def column_tags(df, wl, exceptii=None):
    column_tags = {}
    word_list = [i for sub in wl for i in sub] 
    for column in df.columns:
        if exceptii != None:
            if exceptii.get(column) != None:
                column_tags[column] = exceptii.get(column)
                continue

        column_type = df[column].dtype
        
        if column_type == int and df[column].nunique() < 4:
            if df[column].nunique() == 2:
                column_tags[column] = {'categorie': 'dihotomica', 'subcategorie': 'metrica'}
            else:
                column_tags[column] = {'categorie': 'calitativa', 'subcategorie': 'ordinal'}
        elif column_type == int or column_type == float:
            column_tags[column] = {'categorie': 'cantitativa', 'subcategorie': 'raport' if df[column].min() > 0 else 'interval'}
        else:
            words = any(df[column].str.contains('|'.join(word_list)))
            if words:
                if df[column].nunique() == 2:
                    column_tags[column] = {'categorie': 'dihotomica', 'subcategorie': 'metrica'}
                else:
                    column_tags[column] = {'categorie': 'calitativa', 'subcategorie': 'ordinal'}
            else:
                if df[column].nunique() == 2:
                    column_tags[column] = {'categorie': 'dihotomica', 'subcategorie': 'categoriala'}
                else:
                    column_tags[column] = {'categorie': 'calitativa', 'subcategorie': 'nominala'}
    
    return column_tags

def extract_num(name):
    num = re.search(r'\d+', name)
    if num:
        return int(num.group())
    else:
        return None

def check_sane(df, col):
    n = extract_num(col)
    if n == None:
        return []
    indices = df[df[col] > n].index
    return indices

def remove_outliers(df, col):
    check = True
    while check:
        medie = df[col].mean()
        if tags[col]['subcategorie'] == 'raport':
            var = df[col].var()
            cv = var/medie
            if cv > 0.6:
                check = True
                mediana = df[col].median()
                dif = medie - mediana
                if dif < 0:
                    indx = df[col].idxmin()
                    df = deal_data(df, indx, col)
                else:
                    indx = df[col].idxmax()
                    df = deal_data(df, indx, col)
            else:
                check = False
        else:
            mediana = df[col].median()
            dif = medie - mediana
            if abs(dif) >= 0.01:
                check = True
                if dif < 0:
                    indx = df[col].idxmin()
                    df = deal_data(df, indx, col)
                else:
                    indx = df[col].idxmax()
                    df = deal_data(df, indx, col)
            else:
                check = False
    return df


def find_index(order, strings):
    inds = []
    for s in strings:
        for i, string in enumerate(order):
            if string == s:
                inds.append(i)
    return inds

def encode_column(df, col, order=None):
    categories = df[col].unique()
    if order == None:
        encoded_values = np.arange(len(categories))
    else:
        encoded_values = find_index(order, categories)
    mapping = dict(zip(categories, encoded_values))
    df[col + "_encoded"] = df[col].map(mapping)
    return df, mapping

def decode_column(v, mapp):
    dv = []
    for i in v:
        for key, val in mapp.items():
            if val == i:
                dv.append(key)
                break
    return dv

def describe_num(df, col):
    medie = df[col].mean()
    var = df[col].std()
    mediana = df[col].median()
    mx = df[col].max()
    mn = df[col].min()

    print("Coloana " + col + ":")
    print("Media " + str(medie) + " +/- " + str(var))
    if len(df[col] >= 30):
        interv = scipy.stats.norm.interval(confidence=0.95, loc=medie, scale=var)
    else:
        interv = scipy.stats.t.interval(confidence=0.95, df=(len(df[col])-1) ,loc=medie, scale=var)
    print("Interval de incredere: " + str(interv))

    if tags[col]['subcategorie'] == 'raport':
        cv = var/medie
        print("Coeficientul de variatie: " + str(cv))
    print("Mediana: " +str(mediana) + " cu intervalul valorilor intre [ " + str(mn) + ';' + str(mx) + ' ]')

def describe_cat(df, col):
    ax = sns.histplot(df[col])
    titlu = 'Distributia coloanei "' + col +'"'
    ax.set(title=titlu, ylabel="Numar de raspunsuri")
    plt.show()

    print(col)
    frecv = df[col].value_counts()
    print(frecv.head())

    total = df[col].count()

    
    print(total)
    print(frecv)
    if total > 100:
        rfrecv = frecv/total
        rfrecv = rfrecv.map(lambda x: f"{x:.2%}")
        print(rfrecv)
        
def describe_df(df):
    substr = "_encoded"
    cols = df.columns
    for col in cols:
        if substr in col:
            continue
        if tags[col]['categorie'] == 'cantitativa':
            describe_num(df, col)
        else:
            describe_cat(df, col)
        print()



def corelate(df, col1, col2):
    pearson,_ = scipy.stats.pearsonr(df[col1], df[col2])
    spearman,_ = scipy.stats.spearmanr(df[col1], df[col2])
    if spearman < 0:
        cresc = "descrescatoare"
    else:
        cresc = "crescatoare"
   
    print("Pearson: " + str(pearson))
    print("Spearman: " + str(spearman))

    pear = abs(pearson)
    if pear > 0.9:
        liniaritate = 2
    elif pear > 0.7:
        liniaritate = 1
    elif pear > 0.5:
        liniaritate = 0
    else:
        liniaritate = -1

    if liniaritate < 1:
        mes = "Variabilele nu sunt corelate puternic liniar"
        spear = abs(spearman)
        if spear > 0.9:
            monot = 2
            mes = mes + ", dar totusi prezinta o legatura puternic monoton " + cresc
        elif spear > 0.7:
            monot = 1
            mes = mes + ", dar totusi prezinta o legatura monoton " + cresc

        elif spear > 0.5:
            monot = 0
            mes = mes + " si prezinta o legatura slaba monoton " + cresc
            
        else:
            monot = -1
            mes = mes + " si nici nu se poate afirma cu certitudine ca sunt corelate monoton " + cresc

    else:
        mes = "Variabilele sunt corelate intr-o legatura liniara "
        mes = mes + cresc
    print(mes)

    a,b = np.polyfit(df[col1], df[col2], 1)
    plt.scatter(df[col1], df[col2])
    plt.plot(df[col1], a*df[col1]+b, color='red')
    plt.xlabel(col1)
    plt.ylabel(col2)
    titlu = "Corelatia dintre " + col1 + " si " + col2
    plt.title(titlu)
    plt.show()
    return abs(spearman-pearson)


def test_num(v1, v2, marime, parametru,force=None, normal='shapiro', paired=None):
    if force == 'ttest':
        pass
    n1 = len(v1)
    n2 = len(v2)
    if n1 > 30 and n2 > 30:

        if normal == 'shapiro':
            _, p1 = scipy.stats.shapiro(v1)
            _, p2 = scipy.stats.shapiro(v2)

        else:
            m1 = np.mean(v1)
            var1 = np.std(v1)
            m2 = np.mean(v2)
            var2 = np.std(v2)

            d1 = (v1-m1)/var1
            d2 = (v2-m2)/var2
            _, p1 = scipy.stats.kstest(d1, 'norm')
            _, p2 = scipy.stats.kstest(d2, 'norm')
        if p1 > 0.05 and p2 > 0.05:
            if paired:
                _, pv = scipy.stats.ttest_rel(v1, v2)
            else:
                _, pv = scipy.stats.ttest_ind(v1, v2)
            if pv >= 0.05:
                print('Nu exista diferente semnficative intre mediile lor')
            else:
                print('Exista diferente semnficative intre mediile lor')
            print(f"T-test p-value: {pv:.4f}")
            return
    if paired:
        _, pv = scipy.stats.wilcoxon(v1, v2)
    else:
        _, pv = scipy.stats.mannwhitneyu(v1, v2)
    if pv >= 0.05:
        print('Exista diferente semnficative intre mediile lor')
    else:
        print('Nu exista diferente semnficative intre mediile lor')
    print(f"Mann-Whitney U test p-value: {pv:.4f}")


def test_cat(v1, v2, c1, c2):
    contingency_table = pd.crosstab(v1, v2)
    threshold = 5
    test_table = contingency_table.copy()
    is_valid = (test_table >= threshold).all().all()
    print(contingency_table)
    contingency_table.columns = decode_column(contingency_table.columns, codari[c2]) 
    contingency_table.index = decode_column(contingency_table.index, codari[c1]) 
    print(contingency_table)

    if is_valid:
        pv = scipy.stats.chi2_contingency(contingency_table)[1]
        print("P-value:", pv)
        if pv >= 0.05:
            print('Nu exista o asociere intre cele 2 coloane')
        else:
            print('Exista o asociere intre cele 2 coloane')

    else:
        print("Contingency table does not have enough values for the test.")
   
def test_dih(v1, v2, c1, c2):
    table = pd.crosstab(v1, v2)
    res = scipy.stats.contingency.odds_ratio(table)
    cazuri_expuse = table.loc[1, 1]
    expuse_total = table.loc[1, 1] + table.loc[0, 1]
    cazuri_control = table.loc[1, 0]
    control_total = table.loc[1, 0] + table.loc[0, 0]
    res_risk = scipy.stats.contingency.relative_risk(cazuri_expuse, expuse_total, cazuri_control, control_total)
    interv_or = res.confidence_interval(confidence_level=0.95)
    interv_risk = res_risk.confidence_interval(confidence_level=0.95)
    table.columns = decode_column(table.columns, codari[c2]) 
    table.index = decode_column(table.index, codari[c1]) 
    print(table)
    print('OR:')
    print(res.statistic)
    print(interv_or)
    if interv_or[0] < 1 and 1<interv_or[1]:
        print("Diferentele nu sunt semnificative")

    else:
        print('Diferente semnificative')

    print('RR:')
    print(res_risk.relative_risk)
    print(interv_risk)
    if interv_risk[0] < 1 and 1<interv_risk[1]:
        print("Diferentele nu sunt semnificative")

    else:
        print('Diferente semnificative')


def test_df(df, col1, col2, paired=None, normal='shapiro', ):
    substr1 = '_encoded'
    c1 = col1.replace(substr1, '')
    c2 = col2.replace(substr1, '')
    if tags[c1]['categorie'] == 'cantitativa' and tags[c2]['categorie'] == 'cantitativa':
            test_num(df[col1], df[col2], c1, c2, normal=normal, paired=paired)
    elif tags[c1]['categorie'] == 'calitativa' and tags[c2]['categorie'] == 'calitativa':
        test_cat(df[col1],df[col2], col1, col2)
    elif tags[c1]['categorie'] == 'dihotomica' and tags[c2]['categorie'] == 'dihotomica':
        test_dih(df[col1],df[col2], col1, col2)
    elif tags[c1]['categorie'] == 'cantitativa':
        vals = df[col2].unique()
        vects = []
        for val in vals:
            v = df.loc[df[col2] == val, col1]
            vects.append(v)
        test_num(vects[0], vects[1], c1, c2, paired=paired, normal=normal)
    elif tags[c2]['categorie'] == 'cantitativa':
        vals = df[col1].unique()
        vects = []
        for val in vals:
            v = df.loc[df[col1] == val, col2]
            vects.append(v)
        test_num(vects[0], vects[1], c2, c1, paired=paired, normal=normal)
     
    else:
        print("Wrong")


def create_dih(df, col, val, names):
    substr1 = '_encoded'
    c = col.replace(substr1, '')

    new_col = c + "_dih"
    df[new_col] = df[col].copy()
    df.loc[df[col] <= val, new_col] = 0
    df.loc[df[col] > val, new_col] = 1

    map_dih = dict(zip(names, [0, 1]))

    return df, map_dih

def prepare_data(df, outliers='default', exceptii=None, wl=None):
    cols = df.columns

    #Assign tags
    tags = column_tags(df, wl, exceptii)


    #Deal with null values
    total = 0
    for c in cols:
        inds = no_data(df, c)
        total = total + len(inds)
        while len(inds) != 0:
            df = deal_data(df, inds[0], c)
            inds = no_data(df, c)

    #Deal with wrong values
    total = 0
    for c in cols:
        inds = check_sane(df, c)
        total = total + len(inds)
        while len(inds) != 0:
            df = deal_data(df, inds[0], c)
            inds = no_data(df, c)
        

    #Remove outliers
    if outliers == 'natural':
        pass
    else:
        for c in cols:
            if tags[c]['categorie'] == 'cantitativa':
                df = remove_outliers(df, c)

    #Encode columns
    lst = []
    for c in cols:
        if tags[c]['categorie'] != 'cantitativa':
            lst.append(c)
            if df[c].dtype == int:
                df, mapp = encode_column(df, c)
                codari[c] = mapp
                continue
            for strs in wl:
                inds = find_index(strs, df[c].unique())
                if len(inds) == 0:
                    df, mapp = encode_column(df, c)
                else:
                    df, mapp = encode_column(df, c, order=strs)
                    break
            codari[c] = mapp
    return df, tags, codari




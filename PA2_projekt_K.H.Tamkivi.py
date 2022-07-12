#Palmeri saarestiku pingviinid
#Projekt aines programmeerimise alused II (MTAT.03.256)

#Autor: Karl Hendrik Tamkivi



#Kes kunagi on vähegi kasutanud programmeerimiskeelt R, on ilmselt tuttav ka näidisandmestikuga iris, mis hõlmab endas erinevate iirisesortide õite mõõtmeid. 
#Viimasel ajal on aga laiemalt levima hakanud Antarktika rannikul asuva Palmeri saarestiku pingviinide andmeid sisaldav andmestik, mida samuti heaks õppematerjaliks peetakse. 
#Just seetõttu otsustasin antud aine raames seda populaarsust koguvat andmestikku pisut uurida ja katsetada selle peal kursuse jooksul õpitud Pythoni teadmisi.

#Töös analüüsitav vabavaraline andmestik pärineb kaggle.com veebilehelt ja on allalaetav siit: https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data?resource=download

#Andmestikus sisaldub informatsioon 344 linnu kohta. Iga linnu puhul on registreeritud liik, sugu, mass, noka pikkus, noka sügavus, tiiva pikkus ja saare nimi, kust isend püüti.

#Lühikese projekti eesmärgiks on võrrelda lindude noka proportsionaalseid suurusi võrrelduna kehamassiga. Teisiti öeldes – kui suur on linnu nokk võrreldes tema kehamassiga. 
#Eesmärgiks on omavahel võrrelda kolme erinevat liiki - valjaspingviin (Pygoscelis antarcticus), eeselpingviin (Pygoscelis papua) ja adeelia pingviin (Pygoscelis adeliae)- 
# ning lisaks liikide võrdlemisele uurida ka, kas "nokakoefitsent" erineb ka sugude lõikes. 
#Projekti viimases osas on eesmärgiks lihtsa masinõppe katse abil uurida, kas see sama nokakoefitsent võiks ka reaalsuses olla kolme liigi eristamiseks hea määramistunnus.





#Analüüsiks on vaja mitut erinevat lisapaketti.

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats as st
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


#Järgnevalt loeme sisse andmete .csv faili, mis peaksid asuma samas kaustas, kus on ka koodifail. 10 juhusliku rea näitel saame aimu andmestiku struktuurist.

andmed = pd.read_csv("penguins_size.csv")
print(andmed.sample(10))


#Enne analüüsi tuleks kindlasti kontrollida, kas andmestikus on puuduvaid kirjeid. Näeme, et nii paraku tõesti on. 
#Liik on määratud kõigil isenditel, kuid esineb linde, kel on puudu mõni kehamõõde ja lausa 10 lindu, kelle sugu on määramata.

print(andmed.isnull().sum())


#Pidevate ja nominaalsete andmetega tuleb edasi toimida erinevalt ning võimalikke lahendusi puuduvate kirjete probleemile on mitmeid.
#Pidevate kehamõõtmete puhul on üheks võimaluseks leida kõigi lindude keskmine ja puuduvad kirjed selle keskmisega asendada. Sel puhul on abiks käsk replace().

mean_culmen_length = round(andmed['culmen_length_mm'].mean(),1)
mean_culmen_depth = round(andmed['culmen_depth_mm'].mean(),1)
mean_body_mass = round(andmed['body_mass_g'].mean(),1)

andmed['culmen_length_mm'].replace(np.nan , mean_culmen_length , inplace=True)
andmed['culmen_depth_mm'].replace(np.nan , mean_culmen_depth , inplace=True)
andmed['body_mass_g'].replace(np.nan , mean_body_mass , inplace=True)


#Nominaalsete väärtuste puhul on kõige lihtsamaks lahenduseks vastavad puuduvate väärtustega isendid lihtsalt analüüsist eemaldada. 
#Kindlasti esineb paremaid ja nutikamaid lahendusviise, kuid kuna puuduvate väärtustega linde on vaid 10, ei ole selle meetme efekt liiga drastiline.

#Andmestikku analüüsides selgus veel, et ühe linnu soo väärtuseks oli kirjutatud lihtsalt . , mistõttu tuleb ka see lind eemaldada.

andmed.dropna(subset=['sex'] , axis = 0 , inplace = True)
andmed.drop(andmed.index[andmed['sex'] == '.'], inplace=True)

print(andmed.isnull().sum())

#Protsessi tulemusena oleme puuduvatest väärtustest lahti saanud.


#Järgmiseks eesmärgiks ongi iga isendi jaoks välja arvutada nokakoefitsendi väärtus k.
#k = noka pikkus*noka laius / kehamass

#Selle arvutuse tarbeks loome olemasolevast andmetabelist kaks kahemõõtmelist järjendit. 
#Esimesse lisame iga linnu kohta tema nokamõõtmed ja kehamassi, teise järjendisse liigi ja soo. Esimese järjendi elementidega tehtava tehte väärtuse salvestamegi teise järjendisse.

mõõdud = andmed[['culmen_length_mm', 'culmen_depth_mm', 'body_mass_g']].to_numpy().tolist()
print(mõõdud[1:10])

info_ja_koef = andmed[['species', 'sex']].to_numpy().tolist()
print(info_ja_koef[1:10])

for lind in range(len(mõõdud)) :
    koef = (mõõdud[lind][0]*mõõdud[lind][1])/mõõdud[lind][2]
    info_ja_koef[lind].append(round(koef,3))
print(info_ja_koef[1:10])

#Teeme teise listi omakorda uuesti andmetabeliks ja näeme, et kolme komakohani ümardatud koefitsendi väärtus on lisatud viimasesse tulpa.

uus_tabel = pd.DataFrame(np.array(info_ja_koef))
uus_tabel.columns = ["liik", "sugu", "koef"]
print(uus_tabel.sample(10))


#Et saaksime lõpptulemusi ka visualiseerida, tuleb andmetüüpe veidi manuaalselt modifitseerida.

lõplik = uus_tabel.explode('koef').reset_index(drop=True)
lõplik = lõplik.assign(liik=lõplik['liik'].astype('category'), 
                        sugu=lõplik['sugu'].astype('category'),
                       koef=lõplik['koef'].astype(np.float32))

#Visualiseerime tulemusi karpdiagrammina liikide ja ka sugude kaupa. x-teljele kolm erinevat liiki, iga puhul eraldi emased ja isased isendid, y-teljele nokakoefitsendi väärtus.

joonis = sns.boxplot(x = lõplik['liik'],
            y = lõplik['koef'],
            hue = lõplik['sugu'],
            )
joonis.set(xlabel='liik', ylabel='nokakoefitsent')
#plt.show()


#Joonist vaadates näeme, et nokakoefitsentide väärtused on liikide lõikes visuaalselt üpriski erinevad. Samas näib, et sugude lõike märkimisväärseid erinevusi ei esine. 
#See langeb hästi kokku teooriaga, et pingviinlaste seas üldjuhul sugudevahelist dimorfismi ei esine või on see vaevumärgatav.

#Jooniselt võime välja lugeda, et justkui oleks proportsionaalselt kõige pisemad nokad eeselpingviinidel ja kõige suuremad valjaspingviinidel. 
#Selleks, et seda väidet ka statistiliselt tõestada, viime läbi lühikese liikide keskmisi võrdleva t-testi.

#T-testi läbiviimise jaoks peame esmalt kontrollima, et kas liikide nokakoefitsentide jaotused ühtivad enamvähem normaaljaotusega ning on sarnase hajuvusega. 
#Saame seda teha visuaalselt, kasutades näiteks histogramme. Kontrolliks viime läbi iga liigi siseselt ka normaaljaotust kontrolliva Shapiro testi.


lõplik.groupby('liik').describe()

plt.hist(lõplik['koef'][lõplik['liik']=='Adelie'])
plt.hist(lõplik['koef'][lõplik['liik']=='Chinstrap'])
plt.hist(lõplik['koef'][lõplik['liik']=='Gentoo'])

plt.legend(['Adelie', 'Chinstrap', 'Gentoo'])


shap_a = st.shapiro(lõplik['koef'][lõplik['liik']=='Adelie'])
shap_c = st.shapiro(lõplik['koef'][lõplik['liik']=='Chinstrap'])
shap_g = st.shapiro(lõplik['koef'][lõplik['liik']=='Gentoo'])

print("Adelie - Chinstrap: p-value = " + str(shap_a[1]))
print("Adelie - Gentoo: p-value = " + str(shap_c[1]))
print("Chinstrap - Gentoo: p-value = " + str(shap_g[1]))

#Veendusime, et t-testi eeldused on täidetud - nokakoefitsentide väärtused on nii visuaalselt kui ka statistiliselt normaaljaotusega ja sarnase hajuvusega (p>0,05). 
#Edasi viimegi paarikaupa läbi t-testid.

adelie = lõplik.loc[lõplik['liik'] == 'Adelie', 'koef'].to_numpy()
chinstrap = lõplik.loc[lõplik['liik'] == 'Chinstrap', 'koef'].to_numpy()
gentoo = lõplik.loc[lõplik['liik'] == 'Gentoo', 'koef'].to_numpy()


adelie_chinstrap = st.ttest_ind(a=adelie, b=chinstrap, equal_var=True)
adelie_gentoo = st.ttest_ind(a=adelie, b=gentoo, equal_var=True)
chinstrap_gentoo = st.ttest_ind(a=chinstrap, b=gentoo, equal_var=True)

print("Adelie - Chinstrap: p-value = " + str(adelie_chinstrap[1]))
print("Adelie - Gentoo: p-value = " + str(adelie_gentoo[1]))
print("Chinstrap - Gentoo: p-value = " + str(chinstrap_gentoo[1]))

#Lähtudes t-testi tulemustest võime väita, et 95-protsendilise kindlusega on kõik kolm liiki oma nokakoefitsentidelt teineteisest erinevad ning sisuliselt võiks seda tunnust edaspidi ka liikide eristamiseks kasutada.

#Lubavatest tulemustest innustatuna võimegi antud koefitsendi liigimääramise võime päriselt proovile panna. Korraldame selleks lihtsal masinõppe algoritmil baseeruva katse.

#Eesmärgiks on luua olukord, kus meil on kaks eraldiseisvat andmetabelit - ühes linnud (näiteks 80% koguarvust), kelle andmeid kasutame mudeli "treenimiseks" 
#ja teises linnud (ülejäänud 20%), kelle andmeid kasutame hiljem mudeli headuse testimiseks. 
#Antud mudelil on lihtsakoeline ülesanne: määrata linnud nokakoefitsendi alusel õigesse liiki. 
#Kasutame selleks tuntud k-means algoritmi, mis jagab liigid nokakoefitsendi keskmiste alusel meie soovil kolme klastrisse ehk sisuliselt kolme liiki.

#Antud mudel eeldab, et arvulised väärtused paikneksid kahedimensionaalses järjendis x ja klassitunnus ühedimensionaalses järjendis y.

mudel = KMeans(n_clusters = 3 , random_state = 123)

x = lõplik[['koef']]
y = lõplik['liik']

x_treening,x_test,y_treening,y_test = train_test_split(x, y , test_size = 0.2 , random_state = 123)


#Sobitame mudeli testandmestikule

mudel.fit(x_treening)

#Nüüd on mudel loonud meile ühemõõtmelise järjendi, mis sisaldab väärtusi 0,1 ja 2. 
#Esmapilgul tunduvad need sisutud, kuid tegelikult vastavad need ennustatud liikidele: 0 = eeselpingviin (Gentoo), 1 = valjaspingviin (Chinstrap) ja 2 = adeeliapingviin (Adelie). 
#Meeles tuleb pidada, et tegemist on andmetele sobitatud mudeli ennustustega, mitte mudeli headuse katsetuse tulemustega. Teeme saadud tulemustest lihtsa tabeli.

x_treening['liik'] = y_treening
x_treening['ennustus'] = mudel.labels_

print(x_treening)

#Asendame krüptilised numbrid liiginimedega.

x_treening['ennustus'].replace([0,1,2] , ['Gentoo','Chinstrap','Adelie'] , inplace=True)
print(x_treening)


#Võime mudeli sooritust ka visualiseerida ja arvuliselt hinnata. 

print(pd.crosstab(x_treening['liik'] ,x_treening['ennustus']))

#Näiteks ilmneb risttabelist, et sajaprotsendiliselt mudel ennast andmetele sobitada ei suutnud ja tegi ka valemääranguid. 
#See ei ole tegelikkuses üldse halb märk, kuna me ei soovi mudelit ka ülesobitada. 

tulemus_treening = x_treening['liik'] == x_treening['ennustus']
print(tulemus_treening.mean())

#Arvutades mudeli keskmist hinnangutäpsust, saame treeningandmete puhul tulemuseks 88%.

#Nüüd võime oma mudelit ka reaalsetes oludes testida ehk katsetada seda andmete peal, mida mudel kunagi varem näinud ei ole. 
#Sisuliselt on see justkui pingviini liigi määramine Palmeri saarestikus, kuid meie ainsaks mõõdikuks on nokakoefitsent ja selle väärtustel baseeruv mudel.

ennustus = mudel.predict(x_test)

x_test['liik'] = y_test
x_test['ennustus'] = ennustus

x_test['ennustus'].replace([0,1,2],['Gentoo','Chinstrap','Adelie'],inplace=True)
print(x_test)

#Saame mudeli headust hinnata samal moel, nagu tegime seda ka varem. 

print(pd.crosstab(x_test['liik'] , x_test['ennustus']))

result_test = x_test['liik'] == x_test['ennustus']
print(result_test.mean())

#Näeme, et mudeli ennustustäpsus on üllatavalt kõrge 85%. 
#Seega tõesti, olukorras, kus me teoreetiliselt kolme pingviiniliigi määramistunnustest mitte midagi muud ei tea, võiks nokakoefitsent kolme liigi eristamiseks päris hästi sobida.
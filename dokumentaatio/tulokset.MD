## Tulokset ja anlyysi

# Päätyykö twiitti viraaliksi?

Aineistossa ei ole valmiina selitettävää muutujaa eli onko twiitti viraali vai ei. Selitettävä muuttuja määritetään twiitin uudelleentweettausten määrän perusteella. Rajaksi onko twiitti viraali vai ei otetaan mediaaniluku uudelleen tweettauksista eli 13. Selitettäviksi muuttujiksi määritetään twiitin pituus merkkeinä, käyttäjän seuraajien määrä, käyttäjän ystävien määrä ja se onko twiitissä ollut hashtag (#).

Data skaalataan, jotta K-NN algoritmin ennuste paranisi. Skaalaukseen käytetään scale() funktiota, joka vähentää kolumnin keskiarvon alkuperäisestä arvosta ja jakaa tämän arvon kolumnin keskihajonnalla, jotta luokittelijan ennustettavuutta voidaan arvioida, jaetaan data test- ja train-osajoukkoihin. Testi-joukon kooksi määritetään 20 % koko datasta.

Paras malli saadaan luotua silloin, kun k arvo on 38. Malli luokittelee tällöin 63 % twiiteistä oikein viraaleiksi ja ei viraaleiksi. Pienemmällä k-arvolla malli ylisovittaa datan ja suuremmalla alisovittaa. K-arvo määrittää sen kuinka montaa data-pistettä tietyn pisteen vierestä käytetään luokitteluun.

![kuva k arvon kehityksesta](/kuvat/k_luku.png)

# Onko twiitti peräisin Lontoosta, New Yorkista vai Pariisista?


Datasta yhdistetään ensin kaikkien twiittien tekstit selitettäväksi muuttujaksi ja näille luodaan selitettävä muuttuja kaupunkien (New York, Lontoo, pariisi) perusteella. Teksit muutemaan CountVectorize() funktion avulla vektoreiksi (one-hot enkoodaus). Naive Bayes classifier ennustaa todennäkköisyyden mistä twiitti on peräisin.

Mallin accuracy on 0.68 eli malli ennustaa twiiteistä 68 % oikein. Alhainen tarkkuus on oletettavaa, koska Englanti ja Yhdysvallat ovat englanninkieltä puhuvia maita. Toisaalta confusion matriisin perusteella malli ei juuri koskaan luokittele Pariisista peräisin olevaa twiittiä väärin, mikä on hyvä.

![confusion matrix](/kuvat/con_matrix.png)
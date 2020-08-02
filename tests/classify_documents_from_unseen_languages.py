from pprint import pprint

from models.document_classifier import DocumentClassifier


def main():
    classifier = DocumentClassifier.load()
    prediction = classifier.predict(
        [
            """Barcelona [bərsəlónə] és una ciutat i metròpoli a la costa mediterrània de la península Ibèrica. És la capital de Catalunya,[1] així com de la comarca del Barcelonès i de la província de Barcelona, i la segona ciutat en població i pes econòmic de la península Ibèrica,[2][3] després de Madrid. El municipi creix sobre una plana encaixada entre la serralada Litoral, el mar Mediterrani, el riu Besòs i la muntanya de Montjuïc. La ciutat acull les seus de les institucions d'autogovern més importants de la Generalitat de Catalunya: el Parlament de Catalunya, el President i el Govern de la Generalitat. Pel fet d'haver estat capital del Comtat de Barcelona, rep sovint el sobrenom de Ciutat Comtal. També, com que ha estat la ciutat més important del Principat de Catalunya des d'època medieval, rep sovint el sobrenom o títol de cap i casal.[4]""",
            """Napoli (AFI: /ˈnapoli/[4] ascolta[?·info]; Nápule in napoletano, pronuncia [ˈnɑːpulə] o [ˈnɑːpələ]) è un comune italiano di 962 160 abitanti[2], terzo in Italia per popolazione, capoluogo della regione Campania, dell'omonima città metropolitana e centro di una delle più popolose e densamente popolate aree metropolitane d'Europa denominata «Grande Napoli».[5]

Fondata dai Cumani nell'VIII secolo a.C., fu tra le città più importanti della Magna Grecia[6][7] e giocò un notevole ruolo commerciale, culturale e religioso nei confronti delle popolazioni italiche circostanti[8]. Dopo il crollo dell'Impero romano, nell'VIII secolo la città formò un ducato autonomo indipendente dall'Impero bizantino; in seguito, dal XIII secolo e per più di cinquecento anni, fu capitale del Regno di Napoli; con la Restaurazione divenne capitale del Regno delle Due Sicilie sotto i Borbone fino all'Unità d'Italia.""",
            """Москва́ (произношение (инф.)) — столица России, город федерального значения, административный центр Центрального федерального округа и центр Московской области, в состав которой не входит[5]. Крупнейший по численности населения город России и её субъект — 12 678 079[2] человек (2020), самый населённый из городов, полностью расположенных в Европе, входит в десятку городов мира по численности населения[6], крупнейший русскоязычный город в мире. Центр Московской городской агломерации.

Историческая столица Великого княжества Московского, Русского царства, Российской империи (в 1728—1730 годах), Советской России и СССР. Город-герой. В Москве находятся федеральные органы государственной власти Российской Федерации (за исключением Конституционного суда), посольства иностранных государств, штаб-квартиры большинства крупнейших российских коммерческих организаций и общественных объединений.""",
            """札幌市（さっぽろし）は、北海道の道央地方に位置し、石狩振興局に属する市。道庁所在地および振興局所在地で、政令指定都市である。道の政治・経済・文化の中心地であり、人口約196万人を有する北海道最大の都市である。""",
        ]
    )
    pprint(prediction)
    breakpoint()


if __name__ == "__main__":
    main()

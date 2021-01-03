from naive_bayes.nb_risco import nb_risco_main
from naive_bayes.nb_credit import nb_credit_main
from naive_bayes.nb_census import nb_census_main

from arvores_decisao.ad_risco import ad_risco_main
from arvores_decisao.ad_credit import ad_credit_main
from arvores_decisao.ad_census import ad_census_main

from random_forest.rf_risco import rf_risco_main
from random_forest.rf_credit import rf_credit_main
from random_forest.rf_census import rf_census_main

from svm.svm_credit import svm_credit_main
from svm.svm_census import svm_census_main

def writeLines(text):
    with open('testes_prob.csv','a+') as f:
        f.write(text.replace('.',','))
    f.close()

for i in range(31):
    
    text = str(nb_credit_main(i)) + ';' + str(ad_credit_main(i)) + ';' + str(rf_credit_main(i)) + ';' + str(svm_credit_main(i)) + '\n' if i !=0 else \
        'NaiveBays' + ';' + 'ArvoredeDecisao' + ';' + 'RandomForest' + ';' + 'SVM' + '\n'
    writeLines(text)
    
    # nb_risco_main() + ad_risco_main() + rf_risco_main()
    
    # nb_census_main() +  ad_census_main() + rf_census_main() + svm_census_main()

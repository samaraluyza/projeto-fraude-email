# -*- coding: utf-8 -*-
from __future__ import division

import sys
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from time import time


from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#numero total de data points
n_total = len(data_dict)

print 'Número de total de data points: ', n_total

#alocacao entre classes (POI/non-POI)

n_poi = 0
for pessoa in data_dict:
    if data_dict[pessoa]['poi'] == True:
       n_poi = n_poi + 1

non_poi = n_total - n_poi

print 'Número total de POI: ', n_poi
print 'Número total de pessoão não POI: ' , non_poi

#numero de caracteristicas no data set

all_features = data_dict[data_dict.keys()[0]].keys()
print 'Número total de features no data set: ' , len(all_features)

#existem caracteristicas com muitos valores faltando etc.

data_pandas = pd.DataFrame.from_dict(data_dict, 'index')

print 'Número de NaN por features'

features_name = []
features_proprotion = []

fig, ax = plt.subplots()

for feature in all_features:
    ser = data_pandas[feature].value_counts()
    if ser.index.contains('NaN'):
        features_name.append(feature)
        features_proprotion.append((int(ser['NaN'])/146)*100)


#grafico com a proporção de null por features

x_pos = [i for i, _ in enumerate(features_name)]

plt.bar(x_pos, features_proprotion, color='blue')
plt.xlabel("Atributo")
plt.ylabel("Proporcao de Nulls")
plt.title("Proporcao de nulls por atributo")

plt.xticks(x_pos, features_name, rotation='vertical')

#plt.show()
plt.savefig('atributos.png')
#numero de caracteristicas usadas

### Task 1: Select what features you'll use.

financial_features = ['salary', 'total_payments',  'bonus', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options',  'long_term_incentive', 'restricted_stock'] 
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
poi_feature = ['poi']
features_list = poi_feature + email_features + financial_features

print 'Número total de features usadas: ' , len(features_list)

### Task 2: Remove outliers
fig, ax = plt.subplots()
#Indentify outliers
def plot_salaryXbonus(feature_x, feature_y):
    for person in data_dict:
        feature_1 = data_dict[person][feature_x]
        feature_2 = data_dict[person][feature_y]
        if feature_1!='NaN' and feature_2!='NaN':
            ax.scatter(feature_1, feature_2)
            ax.annotate(person, (feature_1, feature_2))
        else:
            salary = 0
            bonus = 0
            ax.scatter( feature_1, feature_2 )
            ax.annotate(person, (feature_1, feature_2))

    plt.xlabel(feature_x)
    plt.ylabel(feature_y)

    #plt.show()
    plt.savefig('outliner.png')
    #plt.close()

plot_salaryXbonus('salary', 'bonus')




#Dados errados
financial_data_outlines = {}
for person in data_dict:
    financial_data_outlines[person] = 0
    for feature in financial_features:
        if data_dict[person][feature] == "NaN":
            financial_data_outlines[person] = financial_data_outlines[person] + 1

# outline com todos dados financeiros zerados
for person in financial_data_outlines:
    if financial_data_outlines[person] == len(financial_features):
        print 'outline com todos dados financeiros zerados', person
        data_dict.pop(person, 0)


# Remove outliers
data_dict.pop("TOTAL", 0)


### Task 3: Create new feature(s)

def computeFraction(poi_messages, all_messages):
    """ given a number messages to/from POI (numerator) 
    and number of all messages to/from a person (denominator),
    return the fraction of messages to/from that person
    that are from/to a POI
    """
    ### beware of "NaN" when there is no known email address (and so
    ### no filled email features), and integer division!
    ### in case of poi_messages or all_messages having "NaN" value, return 0.
    fraction = 0
    if poi_messages != 'NaN' and all_messages != 'NaN':
        fraction = poi_messages/float(all_messages)

    return fraction

for person in data_dict:

    data_dict[person]["fraction_from_poi"] = computeFraction(data_dict[person]["from_poi_to_this_person"],
                                                         data_dict[person]["from_messages"])
    data_dict[person]["fraction_to_poi"] = computeFraction(data_dict[person]["from_this_person_to_poi"],
                                                       data_dict[person]["to_messages"])




from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

### Store to my_dataset for easy export below.
my_dataset = data_dict


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)





results = []

for i in [5, 10, 'all']:
    Best = SelectKBest(k=i)
    Best.fit_transform(features, labels)
    Result = zip(Best.get_support(), Best.scores_, features_list[1:])
    list_res = list(sorted(Result, key=lambda x: x[1], reverse=True))
    features_list_test = []
    for f in list_res:
        if f[0]==True:
            features_list_test.append(f[2])
    results.append(features_list_test)


#teste número de features 

features_list = poi_feature + results[0]   #5 features
#features_list =  poi_feature + results[1]  # 10 features
#features_list = poi_feature + results[2]   # 15 features




features_list = features_list + ["fraction_from_poi", "fraction_to_poi"]

print features_list

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

scaler = MinMaxScaler()
features = scaler.fit_transform(features)




### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#Aqui será testado qual a quantidade de número de features utilizados 
## split the data to training and testing sets
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


# Provided to give you a starting point. Try a variety of classifiers.

classifiers_dict = {}
classifiers_dict['NaiveBayes'] = {'clf': GaussianNB(),'params': {} }

classifiers_dict['AdaBoost'] = {'clf': AdaBoostClassifier(),'params':  {'n_estimators': [25, 50, 100],
                                                 'algorithm': ['SAMME', 'SAMME.R'],
                                                 'learning_rate': [.2, .5, 1, 1.4, 2.],
                                                 'random_state': [42]} }

classifiers_dict['SVM'] = {'clf': SVC() , 'params': {'kernel': ['poly', 'rbf', 'sigmoid'],'cache_size': [7000],
                             'tol': [0.0001, 0.001, 0.005, 0.05],
                             'decision_function_shape': ['ovo', 'ovr'],
                             'random_state': [42],
                             'verbose' : [False],
                             'C': [100, 1000, 10000]
                             } }


def train(clf, params, features_train, labels_train):  
    #treinar
    clft = GridSearchCV(clf, params)  
    clft = clft.fit(features_train, labels_train) 
    return clft


def valores_avaliacao(pred, labels_test):  
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    total_predictions = 0
    accuracy = 0.
    precision = 0.
    recall = 0.
    f1 = 0.
    f2 = 0.
    ## baseado no arquivo tester.py
    for prediction, truth in zip(pred, labels_test):
        if prediction == 0 and truth == 0:
            true_negatives += 1
        elif prediction == 0 and truth == 1:
            false_negatives += 1
        elif prediction == 1 and truth == 0:
            false_positives += 1
        elif prediction == 1 and truth == 1:
            true_positives += 1
        else:
            print 'Warning: Found a predicted label not == 0 or 1.'
            print 'All predictions should take value 0 or 1.'
            print 'Evaluating performance for processed predictions:'
            break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0 * (true_positives + true_negatives) / total_predictions
        precision = 1.0 * true_positives / (true_positives + false_positives)
        recall = 1.0 * true_positives / (true_positives + false_negatives)
        f1 = 2.0 * true_positives / (2 * true_positives + false_positives + false_negatives)
        f2 = (1 + 2.0 * 2.0) * precision * recall / (4 * precision + recall)
    except:
        print 'Got a divide by zero'
    return accuracy, precision, recall, f1, f2 




    
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)


    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)

#para cada classificador testar valores de avaliação utilizando o classificador com parametros pesquisados por GridSearch
classifiers = classifiers_dict.copy()
for clf in classifiers: 
    t0 = time() 
    classifier = classifiers[clf]['clf']  
    params = classifiers[clf]['params']  

    clft = train(classifier, params, features_train, labels_train)  
    preds = clft.predict(features_test)  

    accuracy, precision, recall, f1, f2 = valores_avaliacao(preds, labels_test)

    classifiers[clf]['clf'] = clft.best_estimator_
    classifiers[clf]['params'] = clft.best_params_
    classifiers[clf]['accuracy'] = accuracy
    classifiers[clf]['precision'] = precision
    classifiers[clf]['recall'] = recall
    classifiers[clf]['f1'] = f1
    classifiers[clf]['f2'] = f2
    classifiers[clf]['n_features'] = len(features_list)
    classifiers[clf]['time'] = (time() - t0)

    

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#Os classificadores já foram otimizados utilizando a função GridSearchCV
# onde os parâmetros são selecionados para se obter uma perfomace melhor

'''

for clf in classifiers:
    print clf
    print classifiers[clf]['clf'] 
    print 'accuracy: ', classifiers[clf]['accuracy'] 
    print 'precision: ', classifiers[clf]['precision'] 
    print 'recall: ', classifiers[clf]['recall'] 
    print 'f1: ', classifiers[clf]['f1'] 
    print 'f2: ', classifiers[clf]['f2']
    print 'n_features: ', classifiers[clf]['n_features']
    print 'time: ', classifiers[clf]['time']
'''   
	
        
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
# Os atributos com maior score são sempre os mesmos
# mesmo testando com diferentes tamanhos de conjutos 



#O classificador com melhor desempenho

clf = classifiers['NaiveBayes']['clf']

print ''

print 'Cross validação com folds = 300'
            
test_classifier(clf, my_dataset, features_list, 300)

print ''

print 'Cross validação com folds = 600'
            
test_classifier(clf, my_dataset, features_list, 600)

print ''

print 'Cross validação com folds = 1000'
            
test_classifier(clf, my_dataset, features_list, 1000)

print ''

print 'Cross validação com folds = 2000'
            
test_classifier(clf, my_dataset, features_list, 2000)

dump_classifier_and_data(clf, my_dataset, features_list)
import pandas as pd
from sklearn import preprocessing
import random

#Lee el csv y elimina data duplicada o vacía
data = pd.read_csv("dataset.csv")
clean_dataset = data.drop_duplicates()
clean_dataset = clean_dataset.dropna()

#Elimina la data de los alumnos que aún no tienen una especialidad
clean_dataset = clean_dataset[clean_dataset.Especialidad_del_momento != 'ING']
clean_dataset = clean_dataset[clean_dataset.Especialidad_del_momento != 'INGA']

#Remueve la data que contenga notas "Aprobado" o "Reprobado"
clean_dataset = clean_dataset[clean_dataset.Nota != 'APR']
clean_dataset = clean_dataset[clean_dataset.Nota != 'REP']

#Transforma a valores númericos las especialidades
labels = clean_dataset['Especialidad_del_momento'].unique().tolist()
mapping = dict(zip(labels,range(1, len(labels)+1)))
clean_dataset.replace({'Especialidad_del_momento': mapping},inplace=True)

#Transforma a valores númericos los periodos
labels_periodo = clean_dataset['Periodo'].unique().tolist()
mapping_periodo = dict(zip(labels_periodo,range(1, len(labels_periodo)+1)))
clean_dataset.replace({'Periodo': mapping_periodo},inplace=True)

#Transforma a valores númericos los cursos
labels_curso = clean_dataset['Codigo_curso'].unique().tolist()
mapping_curso = dict(zip(labels_curso,range(1, len(labels_curso)+1)))
clean_dataset.replace({'Codigo_curso': mapping_curso},inplace=True)

clean_dataset = clean_dataset.reset_index(drop=True)

#Normaliza las notas a valores entre [0,1]
float_array = clean_dataset[['Nota']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
scaled_array = min_max_scaler.fit_transform(float_array)
normalized_dataset = pd.DataFrame(scaled_array)
clean_dataset['Nota'] = normalized_dataset[0]

#Creamos columnas para el diccionario
total_IDs = clean_dataset['ID'].unique()
total_Semesters = clean_dataset['Periodo'].unique()
total_classes = clean_dataset['Codigo_curso'].unique()

#Se guardan los alumnos en un diccionario.
students_dictionary = {}
for index, row in clean_dataset.iterrows():
    if int(row['ID']) not in students_dictionary:
        students_dictionary[int(row['ID'])] = {'esp': row['Especialidad_del_momento'], int(row['Periodo']) : [(row['Codigo_curso'],row['Nota'])]}
    else:
        if int(row['Periodo']) not in students_dictionary[row['ID']]:
            students_dictionary[int(row['ID'])][int(row['Periodo'])]=[(row['Codigo_curso'],row['Nota'])]
        else:
            students_dictionary[int(row['ID'])][int(row['Periodo'])].append((row['Codigo_curso'],row['Nota']))

#Se calcula la cantidad maxima de crusos de un semestre de un alumno            
most_classes_semester = 0
for s in students_dictionary:
    for p in students_dictionary[s]:
        if p != "esp":
            if len(students_dictionary[s][p]) > most_classes_semester:
                most_classes_semester = len(students_dictionary[s][p])

#Separacion de alumnos por especialidad.
INGE={}
INGO={}
INGI={}
INGC={}
for s in students_dictionary:
    if students_dictionary[s]['esp'] == 1:
        INGE[s]=students_dictionary[s]
    elif students_dictionary[s]['esp'] == 2:
        INGO[s]=students_dictionary[s]
    elif students_dictionary[s]['esp'] == 3:
        INGI[s]=students_dictionary[s]
    elif students_dictionary[s]['esp'] == 4:
        INGC[s]=students_dictionary[s]

#Se calcula la especialidad con menor cantidad.
smallest_esp = min(len(INGE),len(INGO),len(INGI),len(INGC))

#Se remueve la diferencia de las especialidades y la especialidad con menor cantidad.
for i in range(len(INGE)-smallest_esp):
    INGE.pop(random.choice(list(INGE.keys())))
for i in range(len(INGO)-smallest_esp):
    INGO.pop(random.choice(list(INGO.keys())))
for i in range(len(INGI)-smallest_esp):
    INGI.pop(random.choice(list(INGI.keys())))
for i in range(len(INGC)-smallest_esp):
    INGC.pop(random.choice(list(INGC.keys())))

#Se vuelvem a unir los los alumnos.       
students_dictionary = {**INGE,**INGO,**INGI,**INGC}

#Headers del dataframe.
headers=['ID','ESP']
for i in range(len(labels_periodo)):
    headers += ['Periodo_%d' % (i+1)]
    for j in range(most_classes_semester):
        headers += ['Curso_%d' % (j+1),'Nota_%d' % (j+1)]

#Dataframe completo de base de datos limpiada y balanceada.
rows = []
for s in students_dictionary:
    row = [s,students_dictionary[s]['esp']]
    sems = len(labels_periodo)
    for key in students_dictionary[s]:
        if key != 'esp':
            row += [key]
            aux_most_classes_semester = most_classes_semester
            for p in students_dictionary[s][key]:
                row += [p[0],p[1]]
                aux_most_classes_semester -= 1
            row += [0,0]*aux_most_classes_semester
            sems -= 1
    for i in range(sems):
        row += [i]
        row += [0]*(most_classes_semester)*2
    rows.append(row)
#Dataframe con todos los datos.
df = pd.DataFrame(rows, columns=headers)
df = df.sort_values(by=['ID'])
df = df.reset_index(drop=True)

#Se separa dataframe por especialidad.
df_inge = df.loc[df['ESP'] == 1]
df_ingo = df.loc[df['ESP'] == 2]
df_ingi = df.loc[df['ESP'] == 3]
df_ingc = df.loc[df['ESP'] == 4]

#Se crean los training, validation y test sets por especialidad.
df_inge_train=df_inge.sample(frac=0.8)
df_inge_validation=df_inge.drop(df_inge_train.index).sample(frac=0.5)
df_inge_test=df_inge.drop(df_inge_train.index).drop(df_inge_validation.index)

df_ingo_train=df_ingo.sample(frac=0.8)
df_ingo_validation=df_ingo.drop(df_ingo_train.index).sample(frac=0.5)
df_ingo_test=df_ingo.drop(df_ingo_train.index).drop(df_ingo_validation.index)

df_ingi_train=df_ingi.sample(frac=0.8)
df_ingi_validation=df_ingi.drop(df_ingi_train.index).sample(frac=0.5)
df_ingi_test=df_ingi.drop(df_ingi_train.index).drop(df_ingi_validation.index)

df_ingc_train=df_ingc.sample(frac=0.8)
df_ingc_validation=df_ingc.drop(df_ingc_train.index).sample(frac=0.5)
df_ingc_test=df_ingc.drop(df_ingc_train.index).drop(df_ingc_validation.index)

#Se unen los training, validation y test sets.
#Dataframe con los datos de training.
df_training = [df_inge_train,df_ingo_train,df_ingi_train,df_ingc_train]
training_set = pd.concat(df_training)
training_set = training_set.sort_values(by=['ID'])
training_set = training_set.reset_index(drop=True)

#Dataframe con los datos de validation.
df_validation = [df_inge_validation,df_ingo_validation,df_ingi_validation,df_ingc_validation]
validation_set = pd.concat(df_validation)
validation_set = validation_set.sort_values(by=['ID'])
validation_set = validation_set.reset_index(drop=True)

#Dataframe con los datos de test.
df_test = [df_inge_test,df_ingo_test,df_ingi_test,df_ingc_test]
test_set = pd.concat(df_test)
test_set = test_set.sort_values(by=['ID'])
test_set = test_set.reset_index(drop=True)

#Se escriben los CSVs de salida.
df.to_csv("complete_dataset.csv", sep=',', encoding='utf-8')
training_set.to_csv("training_set.csv", sep=',', encoding='utf-8')
validation_set.to_csv("validation_set.csv", sep=',', encoding='utf-8')
test_set.to_csv("test_set.csv", sep=',', encoding='utf-8')


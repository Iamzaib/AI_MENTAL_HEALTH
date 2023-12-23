
import re
import numpy as np
import pandas as pd
import pickle as pkl
import streamlit as st
import matplotlib.pyplot as plt
import openai
import seaborn as sns


openai.api_key='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'

def Get_Pkl(pkl_path):
    '''Function to load a pickle file'''
    with open(pkl_path, 'rb') as f:
        scaler = pkl.load(f)
    return scaler

def Get_Label_Maps(map_path):
    '''Function to load a numpy array containing a dictionary'''
    return np.load(map_path, allow_pickle=True).item()

def Load_All_Needed_Pickles():
    '''Function to load all the pickles needed for inference'''
    cost_of_study = Get_Pkl('./Scalers/cost_of_study_Scaler.pkl')
    hours_per_week_university_work = Get_Pkl('./Scalers/hours_per_week_university_work_Scaler.pkl')
    total_device_hours = Get_Pkl('./Scalers/total_device_hours_Scaler.pkl')
    total_social_media_hours = Get_Pkl('./Scalers/total_social_media_hours_Scaler.pkl')
    exercise_per_week = Get_Pkl('./Scalers/exercise_per_week_Scaler.pkl')
    work_hours_per_week = Get_Pkl('./Scalers/work_hours_per_week_Scaler.pkl')
    Dict_Scalers = {
        'cost_of_study': cost_of_study,
        'hours_per_week_university_work': hours_per_week_university_work,
        'total_device_hours': total_device_hours,
        'total_social_media_hours': total_social_media_hours,
        'exercise_per_week': exercise_per_week,
        'work_hours_per_week': work_hours_per_week
    }

    known_disabilities = Get_Pkl('./Label Encoders/known_disabilities_LabelEncoder.pkl')
    stress_in_general = Get_Pkl('./Label Encoders/stress_in_general_LabelEncoder.pkl')
    well_hydrated = Get_Pkl('./Label Encoders/well_hydrated_LabelEncoder.pkl')
    ethnic_group = Get_Pkl('./OneHotEncoders/ethnic_group_OneHotEncoder.pkl')
    home_country = Get_Pkl('./BinaryEncoders/home_country_BinaryEncoder.pkl')
    course_of_study = Get_Pkl('./BinaryEncoders/course_of_study_BinaryEncoder.pkl')
    personality_type = Get_Pkl('./OneHotEncoders/personality_type_OneHotEncoder.pkl')
    family_earning_class = Get_Pkl('./OneHotEncoders/family_earning_class_OneHotEncoder.pkl')
    stress_before_exams = Get_Pkl('./OneHotEncoders/stress_before_exams_OneHotEncoder.pkl')
    timetable_preference = Get_Pkl('./OneHotEncoders/timetable_preference_OneHotEncoder.pkl')
    ts_impact_Grouped = Get_Pkl('./OneHotEncoders/ts_impact_Grouped_OneHotEncoder.pkl')
    year_of_birth = Get_Pkl('./Label Encoders/year_of_birth_LabelEncoder.pkl')
    student_type_location = Get_Pkl('./OneHotEncoders/student_type_location_OneHotEncoder.pkl')
    Dict_LabelEncoders = {
        'known_disabilities': known_disabilities,
        'stress_in_general': stress_in_general,
        'well_hydrated' : well_hydrated,
        'year_of_birth' : year_of_birth
    }

    Dict_BinaryEncoders={
        'home_country' : home_country,
        'course_of_study' : course_of_study
    }

    Dict_OneHotEncoders={
        'ethnic_group' : ethnic_group,
        'personality_type' : personality_type,
        'student_type_location' : student_type_location,
        'family_earning_class':family_earning_class,
        'stress_before_exams':stress_before_exams,
        'timetable_preference':timetable_preference,
        'ts_impact_Grouped':ts_impact_Grouped

    }
    alcohol_consumption = Get_Label_Maps('./Label Maps/alcohol_consumption.npy')
    diet = Get_Label_Maps('./Label Maps/diet.npy')
    quality_of_life = Get_Label_Maps('./Label Maps/quality_of_life.npy')
    year_of_study = Get_Label_Maps('./Label Maps/year_of_study.npy')
    Dict_LabelMaps = {
        'alcohol_consumption': alcohol_consumption,
        'diet': diet,
        'quality_of_life': quality_of_life,
        'year_of_study': year_of_study
    }

    Model = Get_Pkl('Models/UnSMOTE_LGB1_Unoptimized.pkl')

    mental_health_issues_map = Get_Label_Maps('./Label Maps/mental_health_issues_map.npy')

    return Dict_BinaryEncoders,Dict_OneHotEncoders,Dict_Scalers, Dict_LabelEncoders, Dict_LabelMaps, Model

def get_hours(text):
     digits = re.findall(r'\d', text)

     digits = ''.join(digits)
     try:
        digits = int(digits)
     except:
          digits=10


     return digits


def rename_column_name(df):
    # changing the column headers
    column_rename_all = {
         'Response ID':'Response ID',
        'What are the approximate costs for your studies? (tuition fee per year of study, in pound £)' : 'cost_of_study',
        '16. What Country are you from?' : 'home_country',
        '2. What is your ethnic group?' : 'ethnic_group',
        '3. How many hours do you spend on university-related work, separate from your Course Timetable, per week during exams?' : 'hours_per_week_university_work',
        '18. What is your course of study?' : 'course_of_study',
        '6. How would you define your alcohol consumption?' : 'alcohol_consumption',
        '11. Do you have any known disabilities?' : 'known_disabilities',
        '7. Would you consider yourself an introvert or extrovert person? (Definitions from Oxford Languages)' : 'personality_type',
        '17. What is your year of birth?' : 'year_of_birth',
        '9. Would you say that you are normally well hydrated?' : 'well_hydrated',
        '24. How many hours do you spend using technology devices per day (mobile, desktop, laptops, etc)?' : 'total_device_hours',
        '25. How many hours do you spend using social media per day (Instagram, Tiktok, Twitter, etc)?' : 'total_social_media_hours',
        '1. Would you describe your current diet as healthy and balanced?' : 'diet',
        '26. What year of study are you in?' : 'year_of_study',
        '10. How often do you exercise per week?' : 'exercise_per_week',
        '5. How would you define your quality of life? (as defined by the World Health Organization)' : 'quality_of_life',
        '8. In general, do you feel you experience stress while in the University? (tick all that apply)' : 'stress_in_general',
        '12. How many hours per week do you work?' : 'work_hours_per_week',
        'Are you a home student or an international student?' : 'student_type_location',
        'How would you rate your family class? (Family earnings per year)':'family_earning_class',
'Do you feel stressed before exams? (multiple choice)':'stress_before_exams',
    'Do you prefer your timetable to be spread or compact so that you have less stress at university? (e.g. 1-2 busy days or 3-4 days with less lectures)':'timetable_preference',
'23. Do you feel your timetabling structure has any impact on your study, life and health?': 'ts_impact_Grouped'
  }
    
    # regex_pattern = r'^\d+\.\s+'
    # remove_digit = {col: re.sub(regex_pattern, '', col) for col in df.columns}

    # # Rename the columns in the DataFrame
    # df.rename(columns=remove_digit, inplace=True)
    df = df.loc[:, column_rename_all.keys()].rename(columns=column_rename_all)
    #correcting name of home_country

    allowed_values = ['White', 'Black', 'Asian', 'Chinese', 'Arab', 'Mixed']
    timetable_preference_allowed_values=['I prefer my timetable to be compact. (having all my classes in one day or two days in the week)',
       'I prefer my timetable to be spread with long gaps in between classes (eg, 1-2 modules per day, spread over 3 times per week)']
    stress_allowed_values=['Yes (due to employment related issues)',
       'Yes (due to university work), Yes (due to employment related issues), Yes (due to other circumstances such as health, family issues, etc.)',
       'No',
       'Yes (due to other circumstances such as health, family issues, etc.)',
       'Yes (due to university work)',
       'Yes (due to university work), Yes (due to employment related issues)',
       'Yes (due to university work), Yes (due to other circumstances such as health, family issues, etc.)',
       'Yes (combination of two or more of the above)',
       'Yes (due to university work), Yes (due to employment related issues), Yes (due to other circumstances such as health, family issues, etc.), Yes (combination of two or more of the above)',
       'Yes (due to employment related issues), Yes (due to other circumstances such as health, family issues, etc.)',
       'Yes (due to university work), Yes (due to employment related issues), Yes (combination of two or more of the above)',
       'Yes (due to university work), No',
       'Yes (due to university work), Yes (due to other circumstances such as health, family issues, etc.), Yes (combination of two or more of the above)']

    # Filter the DataFrame to keep only rows with values in the 'colA' column that are in the allowed list
    df = df[df['ethnic_group'].isin(allowed_values)]
    df = df[df['timetable_preference'].isin(timetable_preference_allowed_values)]
    df= df[df['stress_before_exams'].isin(stress_allowed_values)]

    nig_name = ['Nigeria','Nig.','Oyo state','nigeria','Nigérian','osun state','NIGERIA','Nigerian']
    ind_name = ['India','Indian']
    uk_name = ['United kingdom','england','U','UK','The UK','Britain','london','Wales','Uk','uk','United Kingdom','England']
    ger_name =['Germany','Deutschland','germany']
    usa_name =['US','The United States','us','United state']
    

    for value in nig_name:
            df.replace({'home_country': {value: 'Nigeria'}}, inplace=True)
    for value in ind_name:
            df.replace({'home_country': {value: 'India'}}, inplace=True)
    for value in uk_name:
            df.replace({'home_country': {value: 'United Kingdom'}}, inplace=True)
    for value in ger_name:
            df.replace({'home_country': {value: 'Germany'}}, inplace=True)
    for value in usa_name:
            df.replace({'home_country': {value: 'United States of America'}}, inplace=True)
    
    #renaming the stress_before_exams 
    value_yes = ['Yes (due to university work)',
             'Yes (due to university work), Yes (due to other circumstances such as health, family issues, etc.)',
             'Yes (due to university work), Yes (due to employment related issues)',
             'Yes (due to employment related issues)',
             'Yes (due to other circumstances, such as health, family issues, etc)',
             'Yes (due to university work), Yes (due to employment related issues), Yes (due to other circumstances such as health, family issues, etc.)',
             'Yes (combination of two or more of the above)',
             'Yes (due to employment related issues), Yes (due to other circumstances such as health, family issues, etc.)',
             'Yes (due to university work), Yes (due to employment related issues), Yes (due to other circumstances such as health, family issues, etc.), Yes (combination of two or more of the above)',
             'Yes (due to university work), Yes (due to employment related issues), Yes (combination of two or more of the above)',
             'Yes (due to university work), Yes (due to other circumstances such as health, family issues, etc.), Yes (combination of two or more of the above)',
             'Yes (due to university work), No'
            
                         ]
    for value in value_yes:
            df.replace({'stress_before_exams': {value: 'Yes'}}, inplace=True)
    # changing the stress in general    
    for value in value_yes:
        df.replace({'stress_in_general': {value: 'Yes'}}, inplace=True)

        #chaning form_of_employment
   #chaning personality_type
    df.personality_type = df.personality_type.replace(['Introvert (a quiet person who is more interested in their own thoughts and feelings than spending time with other people)','Somewhat in between'],'Introvert')
    df.personality_type = df.personality_type.replace(['Extrovert (a lively and confident person who enjoys being with other people)'],'Extrovert')

    # chaning alcohol_consumption
    df.alcohol_consumption = df.alcohol_consumption.replace(["I don't drink alcohol."],'No Drinks')
    df.alcohol_consumption = df.alcohol_consumption.replace(["My alcohol consumption is below moderate"],'Below Moderate')
    df.alcohol_consumption = df.alcohol_consumption.replace(["My alcohol consumption is above moderate."],'Above Moderate')
    df.alcohol_consumption = df.alcohol_consumption.replace(["My alcohol consumption is moderate."],'Moderate')
    df.alcohol_consumption = df.alcohol_consumption.replace(["My alcohol consumption less moderate."],'Less Moderate')

    #chaning diet
    df.diet = df.diet.replace(['Yes, I think my diet is healthy'],'Healthy')
    df.diet = df.diet.replace(['I think my diet is somewhat inbetween'],'Somewhat Inbetween')
    df.diet = df.diet.replace(['No, I think my diet is unhealthy'],'Unhealthy')

   # chaning quality_of_life
    df.quality_of_life = df.quality_of_life.replace(['Medium quality of life.'],'Medium')
    df.quality_of_life = df.quality_of_life.replace(['High quality of life.'],'High')
    df.quality_of_life = df.quality_of_life.replace(['Very high quality of life'],'Very High')
    df.quality_of_life = df.quality_of_life.replace(['Low quality of life.'],'Low')
    df.quality_of_life = df.quality_of_life.replace(['Very low quality of life.'],'Very Low')
    
    df['hours_per_week_university_work']= df['hours_per_week_university_work'].apply(get_hours)
    df['total_device_hours']= df['total_device_hours'].apply(get_hours)
    df['total_social_media_hours']= df['total_social_media_hours'].apply(get_hours)
    df['exercise_per_week']= df['exercise_per_week'].apply(get_hours)
    df['work_hours_per_week']= df['work_hours_per_week'].apply(get_hours)


    
     
    return df

def update_country(country):
    if country in ['France', 'Poland', 'Norway', 'Portugal', 'Albania', 'Mexico']:
        return 'The European'
    elif country == 'United States':
        return 'United States of America'
    else:
        return country
    

def Preprocess_Inference_Dataset(Inference_DF,Dict_BinaryEncoders,Dict_OneHotEncoders,Dict_Scalers,Dict_LabelEncoders,Dict_LabelMaps):
    '''Function to preprocess the inference dataset'''
    Inference_DF = rename_column_name(Inference_DF)
    st.write(Inference_DF)
    for col in Inference_DF.select_dtypes(include=['int']).columns:
        if 'Response' not in col:

            Inference_DF[col] = Dict_Scalers[col].transform(Inference_DF[col].values.reshape(-1,1))

    # st.write(Inference_DF)
    one_hot_columns = ['ethnic_group', "student_type_location", "personality_type",'family_earning_class','stress_before_exams','timetable_preference','ts_impact_Grouped']
    for col in one_hot_columns:
        ohe = Dict_OneHotEncoders[col]
        # if col=='institution_country':
        #      Inference_DF[col] = Inference_DF[col].apply(update_country)
        transformed_data = ohe.transform(np.expand_dims(Inference_DF[col], axis=1)).toarray()
        for i, category in enumerate(ohe.get_feature_names_out([col])):
            Inference_DF[category] = transformed_data[:, i]
        

    binary_columns=['home_country', 'course_of_study']
    for col in binary_columns:
        
        transformed_data = Dict_BinaryEncoders[col].transform(np.expand_dims(Inference_DF[col], axis=1))
        transformed_data=np.array(transformed_data)
       
        for i, category in enumerate(Dict_BinaryEncoders[col].get_feature_names_out([col])):
            Inference_DF[col+'_'+category] = transformed_data[:, i]
    
    Inference_DF.drop(columns=binary_columns,inplace=True)
    Inference_DF.drop(columns=one_hot_columns,inplace=True)
    
    for col in Inference_DF.select_dtypes(include=['object']).columns:
        if col not in ['alcohol_consumption','diet','quality_of_life','year_of_study']:
            for val in Inference_DF[col].unique():
                if val not in Dict_LabelEncoders[col].classes_:
                    Dict_LabelEncoders[col].classes_ = np.append(Dict_LabelEncoders[col].classes_, val)
            Inference_DF[col] = Dict_LabelEncoders[col].transform(Inference_DF[col])

    for col in Inference_DF.select_dtypes(include=['object']).columns:
        if col in ['alcohol_consumption','diet','quality_of_life','year_of_study']:
            Label_Map = Dict_LabelMaps[col]
            for val in Inference_DF[col].unique():
                if val not in Label_Map.keys():
                    Label_Map[val] = len(Label_Map)
            Dict_LabelMaps[col] = Label_Map
            Inference_DF[col] = Inference_DF[col].map(Label_Map)
    
    return Inference_DF

# def Get_Kmeans_Prediction(XTest):
#     '''Function to get the Kmeans prediction'''
#     KMeans_Model = KMeans(n_clusters=2, random_state=0).fit(XTest)
#     Kmeans_Pred = pd.Series(KMeans_Model.predict(XTest)).map({1:'Yes',0:'No'})
#     return Kmeans_Pred

def Get_Model_Prediction(XTest,Model):
    '''Function to get the Model prediction'''
    Model_Pred = pd.Series(Model.predict(XTest)).map({1:'Has Mental Health',0:'Has No Mental Health'})
   
    return Model_Pred,Model.predict(XTest)

def create_combined_plot(df,groups):
    st.header('Combined Plot for All Groups')

    # Create subplots for each group
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))
    fig.subplots_adjust(hspace=0.5)
    axes = axes.flatten()

    for i, (group_name, column_names) in enumerate(groups.items()):
        for col in column_names:
            if df[col].dtype == 'object':
                sns.countplot(data=df, x=col, ax=axes[i])
                axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)
                axes[i].set_title(f'{group_name}: {col} (Count Plot)')
            elif df[col].dtype in ['int64', 'float64']:
                sns.histplot(data=df, x=col, bins=30, ax=axes[i])
                axes[i].set_title(f'{group_name}: {col} (Histogram)')

    st.pyplot(fig)

def FileUploaded(file):
    '''Function to handle the file uploaded by the user'''
    df = None
    if 'csv' in file.name :
        df = pd.read_csv(file)
    elif 'xlsx' in file.name:
        df = pd.read_excel(file)
    
    Dict_BinaryEncoders,Dict_OneHotEncoders, Dict_Scalers, Dict_LabelEncoders, Dict_LabelMaps, Model = Load_All_Needed_Pickles()
    df = Preprocess_Inference_Dataset(df,Dict_BinaryEncoders,Dict_OneHotEncoders,Dict_Scalers,Dict_LabelEncoders,Dict_LabelMaps)

    Xtest = df.drop('Response ID',axis=1)
    # Kmeans_Pred = Get_Kmeans_Prediction(Xtest)
    Results,result_num = Get_Model_Prediction(Xtest,Model)
    # Results = Format_Results(Xtest,Kmeans_Pred,Model_Pred,df)
    st.write(Results)
    Mental = Xtest[result_num == 1]
    Not_Mental = Xtest[result_num == 0]

    
    result_num=list(result_num)
    fig1, ax1 = plt.subplots()
    count_0 = result_num.count(0)
    count_1 = result_num.count(1)
    total_count = len(result_num)

    percent_0 = (count_0 / total_count) * 100
    percent_1 = (count_1 / total_count) * 100

    labels = ['Has No Mental Health', 'Has Mental Health']
    sizes = [percent_0, percent_1]
    colors = ['lightcoral', 'lightskyblue']
    explode = (0.1, 0)  # Explode the 0 slice
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
             startangle=90,colors=colors)
    ax1.axis('equal')  

    st.pyplot(fig1)
    demography_cols = ['ethnic_group_Arab',"ethnic_group_Asian","ethnic_group_Black","ethnic_group_Chinese","ethnic_group_Mixed","ethnic_group_White", 'year_of_birth', df.columns[21],df.columns[22],df.columns[23]]
    demography_df = df[demography_cols]
    # create_combined_plot(demography_df,groups)
    st.header('Correlation Heatmap for Demography Columns')
    plt.figure(figsize=(10,5))
    sns.heatmap(demography_df.corr(), annot=True, cmap='coolwarm',vmin=1,vmax=2)
    plt.axis('off')
    plt.title('Demography Correlations')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    columns_to_plot = [
        'family_earning_class_Higher class',
        'family_earning_class_Lower class',
        'family_earning_class_Middle class',
        'home_country_0_0',
        'home_country_0_1',
        'home_country_0_2',
        'home_country_0_3',
        'home_country_0_4',
        'course_of_study_0_0',
        'course_of_study_0_1',
        'course_of_study_0_2',
        'course_of_study_0_3',
        'course_of_study_0_4',
        'course_of_study_0_5',
        'course_of_study_0_6',
        'course_of_study_0_7',
        'stress_before_exams_No',
        'stress_before_exams_Yes',
        'timetable_preference_I prefer my timetable to be compact. (having all my classes in one day or two days in the week)',
        'timetable_preference_I prefer my timetable to be spread with long gaps in between classes (eg, 1-2 modules per day, spread over 3 times per week)'
    ]

    # Create a bar plot for each column
    st.header('Bar Plot For Daily activities including work and academic activities,')
    
    plt.figure(figsize=(12, 6))
    daily_activity_df = df[columns_to_plot]
    # Group the columns into lists
    family_earnings = [
        'family_earning_class_Higher class',
        'family_earning_class_Lower class',
        'family_earning_class_Middle class'
    ]

    home_country = [
        'home_country_0_0',
        'home_country_0_1',
        'home_country_0_2',
        'home_country_0_3',
        'home_country_0_4'
    ]

    course_study = [
        'course_of_study_0_0',
        'course_of_study_0_1',
        'course_of_study_0_2',
        'course_of_study_0_3',
        'course_of_study_0_4',
        'course_of_study_0_5',
        'course_of_study_0_6',
        'course_of_study_0_7'
    ]

    stress_before_exams = [
        'stress_before_exams_No',
        'stress_before_exams_Yes'
    ]

    timetable_preference = [
        'timetable_preference_I prefer my timetable to be compact. (having all my classes in one day or two days in the week)',
        'timetable_preference_I prefer my timetable to be spread with long gaps in between classes (eg, 1-2 modules per day, spread over 3 times per week)'
    ]

    group_name=['Family Earnings','Home Country','Course Study','Stress Before Exams','TimeTable Preference']

    for i, group in enumerate([family_earnings, home_country, course_study, stress_before_exams, timetable_preference], start=1):
        fig, ax = plt.subplots(figsize=(8, 6))
        df_group = daily_activity_df[group]
        df_group.sum().plot(kind='bar', ax=ax)
        
        plt.title(f'{group_name[i-1]} Bar Plot')
        plt.xlabel('Column')
        plt.ylabel('Count')
        plt.axis('off')
        plt.xticks(rotation=45)
        st.pyplot()




            

def RecordInputted(record_id):
    '''Function to handle the record id inputted by the user'''
    df = pd.read_csv('Wellbeingdata_yr2223.csv')
    df = df[df['Response ID'] == record_id]

    Dict_BinaryEncoders,Dict_OneHotEncoders, Dict_Scalers, Dict_LabelEncoders, Dict_LabelMaps, Model = Load_All_Needed_Pickles()
    df = Preprocess_Inference_Dataset(df,Dict_BinaryEncoders,Dict_OneHotEncoders,Dict_Scalers,Dict_LabelEncoders,Dict_LabelMaps)
    if len(df) == 0: #if the record id is not found in the inference dataset
        st.write("Record ID not found. Hence Test Data is not available.")
        return

    Xtest = df.drop('Response ID',axis=1)

    Kmeans_Pred = None
    # if len(df) >= 2:
    #     Kmeans_Pred = Get_Kmeans_Prediction(Xtest)
    # Model_Pred = Get_Model_Prediction(Xtest,Model)
    # Results = Format_Results(Xtest,Kmeans_Pred,Model_Pred,df)
    Results,result_num = Get_Model_Prediction(Xtest,Model)
    
    if result_num==0:
        placeholder='Has No Mental Heatlh'
        st.write('Has No Mental Heatlh')
    else:
         placeholder='Has Mental Heatlh'
         st.write('Has Mental Heatlh')
         
    st.header('Mental health recommendations for Higher Education Students')
    with st.spinner('Processing.....'):
        response = openai.Completion.create(
                model='text-davinci-003',
                prompt=f'''
            "A higher education student in term-time has been predicted 
            {placeholder}; 
            the model was trained on student historical data including Demography, 
            Exposure to modern devices and social media, Daily activities including 
            work and academic activities, Lifestyle and healthy living, Family background 
            and social class, Residency and location and History of mental health and disabilities. 
            Using number points advise and give some recommendations to the student.
                ''',
                n=1,
                max_tokens=500
            )
    st.success(response['choices'][0]['text'])
    

def main():
    '''Main function'''
    upload_file = st.radio("How would you like to input your data?", ["Upload File", "Enter Record ID"])
    if upload_file == "Upload File":
        file = st.file_uploader("Upload your file")
        if file is not None:
            FileUploaded(file)
    else:
        record_id = -1
        record_id = st.number_input("Enter Record ID",value=-1)
        if record_id != -1:
            RecordInputted(record_id)

if __name__ == "__main__":
    main()

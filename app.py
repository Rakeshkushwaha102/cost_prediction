#import jsonify

import pickle

import streamlit as st


pickle_in = open("model.pkl", 'rb')
final_model = pickle.load(pickle_in)

def main():
    
    html_temp = """
    <div style="background-color:SkyBlue;padding:5px">
    <h2 style="color:white;text-align:center;">Cost Prediction of New Product in Ed-Tech Industry </h2>
    </div>
    """
    st.title('Ed-Tech ML App')
    st.markdown(html_temp,unsafe_allow_html=True)
    
    Infrastructure=st.number_input("Enter the infrastructure Cost")
    
    internet_Electricity=st.number_input("Enter the internet_Electricity Cost")
    
    nonTeaching_staff_salary=st.number_input("Enter the nonTeaching_staff_salary Cost")
    
    marketing_expenses=st.number_input("Enter the marketing_expenses Cost")
    
    no_of_hours=st.number_input("Enter the no of hours of learning content")
    
    rent_replaced=st.number_input("Enter the rent Cost")
    
    teaching_staff_salary_replaced=st.number_input("Enter the teaching staff salary Cost")
    
    expenditure_of_maintanence_replaced=st.number_input("Enter the expenditure of maintanence Cost")
    
    
    Trainer1= st.selectbox("Enter the mode of course if course?", ['Offline','Online'])
    if(Trainer1 =='Offline'):
        mode_of_course_offline=1
    else:
        mode_of_course_offline = 0
    if(Trainer1 =='Online'):
        mode_of_course_online=1
    else:
        mode_of_course_online = 0 
            
            
    Trainer2= st.selectbox("Is webaccess  provided", ['Yes', 'No'])
    if(Trainer2 =='Yes'):
        webaccess_yes = 1
    else:
        webaccess_yes = 0
    if(Trainer2 =='No'):
        webaccess_no = 1
    else:
        webaccess_no = 0    
    Trainer3= st.selectbox("Is internship_certificates  provided", ['Yes', 'No'])
    if(Trainer3 =='Yes'):
        internship_certificates_yes = 1
    else:
        internship_certificates_yes = 0
    if(Trainer2 =='No'):
        internship_certificates_no = 1
    else:
        internship_certificates_no = 0 
    Trainer4= st.selectbox("Is books supplies  provided", ['Yes', 'No'])
    if(Trainer4 =='Yes'):
        books_supplies_Yes = 1
    else:
        books_supplies_Yes = 0
    if(Trainer4 =='No'):
        books_supplies_No = 1
    else:
        books_supplies_No = 0 
    
    Trainer5= st.selectbox("What is courses category", ['AI', 'Data_Science','Java','Python','SQL'])
    if(Trainer5 =='AI'):
        courses_AI = 1
    else:
        courses_AI = 0
    if(Trainer5 =='Data_Science'):
        courses_Data_Science = 1
    else:
        courses_Data_Science = 0 
    if(Trainer5 =='Java'):
        courses_Java = 1
    else:
        courses_Java = 0 
    if(Trainer5 =='Python'):
        courses_Python = 1
    else:
        courses_Python = 0 
    if(Trainer5 =='SQL'):
        courses_SQL = 1
    else:
        courses_SQL = 0 
    Trainer6= st.selectbox("What is level of course", ['Advance', 'Beginner','Intermediate'])
    if(Trainer6 =='Advance'):
        level_of_course_Advance = 1
    else:
        level_of_course_Advance = 0  
    if(Trainer6 =='Beginner'):
        level_of_course_Beginner = 1
    else:
        level_of_course_Beginner = 0 
    if(Trainer6 =='Intermediate'):
        level_of_course_Intermediate = 1
    else:
        level_of_course_Intermediate = 0     

    internship_certificates_No=0

    result=""
    if st.button("Predict"):
        result= final_model.predict([[Infrastructure,internet_Electricity,nonTeaching_staff_salary,marketing_expenses,no_of_hours,rent_replaced,teaching_staff_salary_replaced,expenditure_of_maintanence_replaced,mode_of_course_offline,mode_of_course_online,webaccess_no,webaccess_yes,internship_certificates_No,internship_certificates_no,internship_certificates_yes,books_supplies_No,books_supplies_Yes,courses_AI,courses_Data_Science,courses_Java,courses_Python,courses_SQL,level_of_course_Advance,level_of_course_Beginner,level_of_course_Intermediate]])
        output =result
        st.success('The product price is {}'.format(output))
   
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")





if __name__=="__main__":
    main()
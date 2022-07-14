import pickle

import streamlit as st
s = r"‪C:\Users\saiku\model.pkl"
s = s.lstrip('\u202a')
pickle_in = open(s, 'rb')
final_model = pickle.load(pickle_in)

def main():
    
    st.title('Edtech Price')
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Edtech ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    #Platform_Institute = st.text_input("Platform_Institute")
    #Coursename = st.text_input("Coursename")
    Infrastructure=st.number_input("Enter the infrastructure Cost")
    
    internet_Electricity=st.number_input("Enter the internet_Electricity Cost")
    
    nonTeaching_staff_salary=st.number_input("Enter the nonTeaching_staff_salary Cost")
    
    marketing_expenses=st.number_input("Enter the marketing_expenses Cost")
    
    no_of_hours=st.number_input("Enter the no of hours of learning content")
    
    rent_replaced=st.number_input("Enter the rent Cost")
    
    teaching_staff_salary_replaced=st.number_input("Enter the teaching staff salary Cost")
    
    expenditure_of_maintanence_replaced=st.number_input("Enter the expenditure of maintanence Cost")
    
    mode_of_course_offline=st.number_input("Enter the mode of course if course is offline enter 1 otherwise enter 0")
    mode_of_course_online=st.number_input("Enter the mode of course if course is online enter 1 otherwise enter 0")
    webaccess_yes=st.number_input("If webaccess is  provided enter 1 otherwise enter 0")
    webaccess_no=st.number_input("If webaccess is not provided enter 1 otherwise enter 0")
    internship_certificates_No=1
    internship_certificates_no=st.number_input("If internship_certificates is not provided enter 1 otherwise enter 0")
    internship_certificates_yes=st.number_input("If internship_certificates is provided enter 1 otherwise enter 0")
    books_supplies_No=st.number_input("If books supplies is not provided enter 1 otherwise enter 0")
    books_supplies_Yes=st.number_input("If books supplies is  provided enter 1 otherwise enter 0")
    courses_AI=st.number_input("If courses category is AI enter 1 otherwise enter 0")
    courses_Data_Science=st.number_input("If courses category is Data_Science enter 1 otherwise enter 0")
    courses_Java=st.number_input("If  courses category is Java enter 1 otherwise enter 0")
    courses_Python=st.number_input("If courses category is Python enter 1 otherwise enter 0")
    courses_SQL=st.number_input("If courses category is SQL enter 1 otherwise enter 0")
    level_of_course_Advance=st.number_input("If level of course is Advance enter 1 otherwise enter 0")
    level_of_course_Beginner=st.number_input("If level of course is Beginner enter 1 otherwise enter 0")
    level_of_course_Intermediate=st.number_input("If level of course is Intermediate enter 1 otherwise enter 0")
    Infrastructure = (Infrastructure-50000)/(1490000-50000)
    internet_Electricity = (internet_Electricity-10000)/(50000-10000)
    nonTeaching_staff_salary = (nonTeaching_staff_salary-8000)/(16000-8000)
    marketing_expenses = (marketing_expenses-2000)/(8000-2000)
    no_of_hours = (no_of_hours-80)/(240-80)
    rent_replaced = (rent_replaced-1500)/(18000-1500)
    teaching_staff_salary_replaced = (teaching_staff_salary_replaced-16000)/(57000-16000)
    expenditure_of_maintanence_replaced = (expenditure_of_maintanence_replaced-10000)/(39000-10000)
    
    
    
    
    
    
    
    
    
    

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

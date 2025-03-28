import numpy as np
import nltk
import string
import os
import sys
import traceback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from flask import Flask, request, jsonify
from flask_cors import CORS

# Download NLTK data at startup
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

# Training data structure: (user_input, category, expense_type)
training_data = [

    # Bills & Utilities entries
    ("Meralco bill payment", "Bills & Utilities", "Electricity"),
    ("Monthly electricity charge", "Bills & Utilities", "Electricity"),
    ("Kuryente bill at home", "Bills & Utilities", "Electricity"),
    ("Power meter payment", "Bills & Utilities", "Electricity"),
    ("Electricity utility bill", "Bills & Utilities", "Electricity"),
    ("Bayad kuryente sa buwan", "Bills & Utilities", "Electricity"),
    ("Electric company invoice", "Bills & Utilities", "Electricity"),
    ("Residential power bill", "Bills & Utilities", "Electricity"),
    ("Kontador ng kuryente", "Bills & Utilities", "Electricity"),
    ("Electric service charge", "Bills & Utilities", "Electricity"),
    ("kuryente payment", "Bills & Utilities", "Electricity"),
    ("electricity bill", "Bills & Utilities", "Electricity"),
    ("power bill payment", "Bills & Utilities", "Electricity"),
    ("utility bill electric", "Bills & Utilities", "Electricity"),
    ("bayad kuryente", "Bills & Utilities", "Electricity"),
    ("bayad ilaw", "Bills & Utilities", "Electricity"),
    ("kontador kuryente", "Bills & Utilities", "Electricity"),
    
    ("Maynilad water bill", "Bills & Utilities", "Water"),
    ("Manila Water payment", "Bills & Utilities", "Water"),
    ("tubig bill", "Bills & Utilities", "Water"),
    ("water bill payment", "Bills & Utilities", "Water"),
    ("monthly water utilities", "Bills & Utilities", "Water"),
    ("bayad tubig", "Bills & Utilities", "Water"),
    ("bill ng tubig", "Bills & Utilities", "Water"),
    ("water meter charge", "Bills & Utilities", "Water"),
    ("water service fee", "Bills & Utilities", "Water"),
    ("water utility bill", "Bills & Utilities", "Water"),
    ("water bill", "Bills & Utilities", "Water"),

    ("PLDT Home Fiber monthly", "Bills & Utilities", "Internet"),
    ("Globe At Home internet", "Bills & Utilities", "Internet"),
    ("Converge ICT payment", "Bills & Utilities", "Internet"),
    ("internet monthly bill", "Bills & Utilities", "Internet"),
    ("wifi service payment", "Bills & Utilities", "Internet"),
    ("broadband payment", "Bills & Utilities", "Internet"),
    ("bayad internet", "Bills & Utilities", "Internet"),
    ("wifi bill", "Bills & Utilities", "Internet"),
    ("load sa wifi", "Bills & Utilities", "Internet"),

    ("Globe postpaid plan", "Bills & Utilities", "Phone Plan"),
    ("Smart postpaid bill", "Bills & Utilities", "Phone Plan"),
    ("prepaid load Globe", "Bills & Utilities", "Phone Plan"),
    ("Smart regular load", "Bills & Utilities", "Phone Plan"),
    ("TnT load", "Bills & Utilities", "Phone Plan"),
    ("mobile plan monthly", "Bills & Utilities", "Phone Plan"),
    ("phone bill payment", "Bills & Utilities", "Phone Plan"),
    ("bayad telepono", "Bills & Utilities", "Phone Plan"),
    ("load sa phone", "Bills & Utilities", "Phone Plan"),
    ("Gomo load", "Bills & Utilities", "Phone Plan"),
    ("Dito load", "Bills & Utilities", "Phone Plan"),
    ("Talk load", "Bills & Utilities", "Phone Plan"),
    ("TNT load", "Bills & Utilities", "Phone Plan"),
    ("Talk N Text load", "Bills & Utilities", "Phone Plan"),
    ("Sun Cellular load", "Bills & Utilities", "Phone Plan"),
    ("Sun load", "Bills & Utilities", "Phone Plan"),
    ("TM load", "Bills & Utilities", "Phone Plan"),

    ("Netflix subscription", "Bills & Utilities", "Streaming Services"),
    ("Spotify Premium", "Bills & Utilities", "Streaming Services"),
    ("Disney+ subscription", "Bills & Utilities", "Streaming Services"),
    ("Amazon Prime", "Bills & Utilities", "Streaming Services"),
    ("HBO Go monthly", "Bills & Utilities", "Streaming Services"),
    ("Viu Premium", "Bills & Utilities", "Streaming Services"),
    ("YouTube Premium", "Bills & Utilities", "Streaming Services"),
    ("Apple Music subscription", "Bills & Utilities", "Streaming Services"),
    ("WeTV subscription", "Bills & Utilities", "Streaming Services"),
    ("iQiyi monthly", "Bills & Utilities", "Streaming Services"),
    ("streaming service monthly", "Bills & Utilities", "Streaming Services"),
    ("video streaming subscription", "Bills & Utilities", "Streaming Services"),
    ("music subscription", "Bills & Utilities", "Streaming Services"),
    ("bayad Netflix", "Bills & Utilities", "Streaming Services"),
    ("bayad Spotify premium", "Bills & Utilities", "Streaming Services"),
    ("bayad Disney+", "Bills & Utilities", "Streaming Services"),
    ("bayad Amazon Prime", "Bills & Utilities", "Streaming Services"),
    ("bayad HBO Go", "Bills & Utilities", "Streaming Services"),
    ("bayad Viu Premium", "Bills & Utilities", "Streaming Services"),
    ("bayad YouTube Premium", "Bills & Utilities", "Streaming Services"),
    ("bayad Apple Music", "Bills & Utilities", "Streaming Services"),
    ("bayad WeTV", "Bills & Utilities", "Streaming Services"),
    ("bayad iQiyi", "Bills & Utilities", "Streaming Services"),
    ("bayad streaming service", "Bills & Utilities", "Streaming Services"),
    ("bayad video streaming", "Bills & Utilities", "Streaming Services"),
    ("bayad music subscription", "Bills & Utilities", "Streaming Services"),

    ("Monthly rent BGC condo", "Bills & Utilities", "Rent/Mortgage"),
    ("House rental Quezon City", "Bills & Utilities", "Rent/Mortgage"),
    ("Condo association dues", "Bills & Utilities", "Rent/Mortgage"),
    ("rent payment apartment", "Bills & Utilities", "Rent/Mortgage"),
    ("house amortization", "Bills & Utilities", "Rent/Mortgage"),
    ("pag-ibig housing loan", "Bills & Utilities", "Rent/Mortgage"),
    ("monthly association dues", "Bills & Utilities", "Rent/Mortgage"),
    ("annual condo dues", "Bills & Utilities", "Rent/Mortgage"),
    ("renta bahay", "Bills & Utilities", "Rent/Mortgage"),
    ("upa ng apartment", "Bills & Utilities", "Rent/Mortgage"),
    ("monthly apartment rent", "Bills & Utilities", "Rent/Mortgage"),
    ("house payment", "Bills & Utilities", "Rent/Mortgage"),
    ("monthly mortgage", "Bills & Utilities", "Rent/Mortgage"),
    ("monthly rent", "Bills & Utilities", "Rent/Mortgage"),
    ("condo", "Bills & Utilities", "Rent/Mortgage"),
    ("renta", "Bills & Utilities", "Rent/Mortgage"),
    ("upahan", "Bills & Utilities", "Rent/Mortgage"),
    ("bahay", "Bills & Utilities", "Rent/Mortgage"),
    ("apartment", "Bills & Utilities", "Rent/Mortgage"),
    ("renta ng bahay", "Bills & Utilities", "Rent/Mortgage"),
    ("upa ng bahay", "Bills & Utilities", "Rent/Mortgage"),
    ("condominium", "Bills & Utilities", "Rent/Mortgage"),

    ("cooking gas refill", "Bills & Utilities", "Gas"),
    ("gas utility bill", "Bills & Utilities", "Gas"),
    ("Petron gas tank refill", "Bills & Utilities", "Gas"),
    ("Shell gas for stove", "Bills & Utilities", "Gas"),
    ("LPG gas delivery", "Bills & Utilities", "Gas"),
    ("bayad gas", "Bills & Utilities", "Gas"),
    ("tangke ng gas", "Bills & Utilities", "Gas"),
    ("gasul", "Bills & Utilities", "Gas"),
    ("LPG", "Bills & Utilities", "Gas"),
    ("gasolina", "Bills & Utilities", "Gas"),
    ("gasul delivery", "Bills & Utilities", "Gas"),
    ("gasul refill", "Bills & Utilities", "Gas"),
    ("gasul tank", "Bills & Utilities", "Gas"),
    ("gasul stove", "Bills & Utilities", "Gas"),
    ("gasul tank refill", "Bills & Utilities", "Gas"),
    ("gasul delivery", "Bills & Utilities", "Gas"),
    ("gasul refill", "Bills & Utilities", "Gas"),
    ("gasul tank", "Bills & Utilities", "Gas"),
    ("gasul stove", "Bills & Utilities", "Gas"),
    ("gasul tank refill", "Bills & Utilities", "Gas"),


    # Food & Groceries entries
    ("SM Supermarket grocery", "Food & Groceries", "Groceries"),
    ("Puregold weekly groceries", "Food & Groceries", "Groceries"),
    ("Robinson's Supermarket", "Food & Groceries", "Groceries"),
    ("Marketplace by Rustan's", "Food & Groceries", "Groceries"),
    ("Landers shopping", "Food & Groceries", "Groceries"),
    ("S&R grocery shopping", "Food & Groceries", "Groceries"),
    ("MetroMart groceries delivery", "Food & Groceries", "Groceries"),
    ("palengke groceries", "Food & Groceries", "Groceries"),
    ("wet market", "Food & Groceries", "Groceries"),
    ("sari-sari store", "Food & Groceries", "Groceries"),
    ("7-Eleven snacks", "Food & Groceries", "Groceries"),
    ("supermarket shopping", "Food & Groceries", "Groceries"),
    ("weekly grocery run", "Food & Groceries", "Groceries"),
    ("food supplies", "Food & Groceries", "Groceries"),
    ("fresh produce", "Food & Groceries", "Groceries"),
    ("meat and poultry", "Food & Groceries", "Groceries"),
    ("pamibili sa palengke", "Food & Groceries", "Groceries"),
    ("grocery sa supermarket", "Food & Groceries", "Groceries"),
    ("mga bigas", "Food & Groceries", "Groceries"),
    ("ulam", "Food & Groceries", "Groceries"),
    ("pagkain", "Food & Groceries", "Groceries"),
    ("gulay", "Food & Groceries", "Groceries"),
    ("karne", "Food & Groceries", "Groceries"),
    ("prutas", "Food & Groceries", "Groceries"),
    ("paninda", "Food & Groceries", "Groceries"),
    ("pagkain sa bahay", "Food & Groceries", "Groceries"),
    ("pagkain sa tindahan", "Food & Groceries", "Groceries"),
    ("pagkain sa palengke", "Food & Groceries", "Groceries"),
    ("pagkain sa supermarket", "Food & Groceries", "Groceries"),

    ("Jollibee lunch", "Food & Groceries", "Dining Out"),
    ("McDonald's breakfast", "Food & Groceries", "Dining Out"),
    ("KFC dinner", "Food & Groceries", "Dining Out"),
    ("Chowking merienda", "Food & Groceries", "Dining Out"),
    ("Max's Restaurant", "Food & Groceries", "Dining Out"),
    ("Ministop food", "Food & Groceries", "Dining Out"),
    ("Uncle John's food", "Food & Groceries", "Dining Out"),
    ("Mang Inasal", "Food & Groceries", "Dining Out"),
    ("Goldilocks", "Food & Groceries", "Dining Out"),
    ("Greenwich pizza", "Food & Groceries", "Dining Out"),
    ("breakfast meal", "Food & Groceries", "Dining Out"),
    ("lunch restaurant", "Food & Groceries", "Dining Out"),
    ("dinner out", "Food & Groceries", "Dining Out"),
    ("afternoon snack", "Food & Groceries", "Dining Out"),
    ("almusal sa labas", "Food & Groceries", "Dining Out"),
    ("tanghalian", "Food & Groceries", "Dining Out"),
    ("hapunan sa labas", "Food & Groceries", "Dining Out"),
    ("meryenda", "Food & Groceries", "Dining Out"),
    ("kainan sa labas", "Food & Groceries", "Dining Out"),


    ("Starbucks coffee", "Food & Groceries", "Coffee/Beverages"),
    ("CBTL drinks", "Food & Groceries", "Coffee/Beverages"),
    ("Tim Hortons coffee", "Food & Groceries", "Coffee/Beverages"),
    ("Dunkin Donuts", "Food & Groceries", "Coffee/Beverages"),
    ("Ministop coffee", "Food & Groceries", "Coffee/Beverages"),
    ("Uncle John's coffee", "Food & Groceries", "Coffee/Beverages"),
    ("coffee shop", "Food & Groceries", "Coffee/Beverages"),
    ("milk tea", "Food & Groceries", "Coffee/Beverages"),
    ("bubble tea", "Food & Groceries", "Coffee/Beverages"),
    ("kape sa labas", "Food & Groceries", "Coffee/Beverages"),
    ("milk tea shop", "Food & Groceries", "Coffee/Beverages"),
    ("bubble tea shop", "Food & Groceries", "Coffee/Beverages"),
    ("kape", "Food & Groceries", "Coffee/Beverages"),
    ("milk tea", "Food & Groceries", "Coffee/Beverages"),
    ("cafe", "Food & Groceries", "Coffee/Beverages"),

    ("GrabFood delivery", "Food & Groceries", "Food Delivery"),
    ("Foodpanda order", "Food & Groceries", "Food Delivery"),
    ("Pick.A.Roo food delivery", "Food & Groceries", "Food Delivery"),
    ("Yellow Cab delivery", "Food & Groceries", "Food Delivery"),
    ("Angel's Pizza", "Food & Groceries", "Food Delivery"),
    ("food delivery order", "Food & Groceries", "Food Delivery"),
    ("online food order", "Food & Groceries", "Food Delivery"),
    ("meal delivery", "Food & Groceries", "Food Delivery"),
    ("delivery ng pagkain", "Food & Groceries", "Food Delivery"),
    ("padeliver na food", "Food & Groceries", "Food Delivery"),
    ("order sa foodpanda", "Food & Groceries", "Food Delivery"),
    ("order sa grabfood", "Food & Groceries", "Food Delivery"),
    ("order sa pickaroo", "Food & Groceries", "Food Delivery"),
    ("order sa delivery", "Food & Groceries", "Food Delivery"),
    ("order sa food delivery", "Food & Groceries", "Food Delivery"),
    ("order sa online food", "Food & Groceries", "Food Delivery"),
    ("order sa meal delivery", "Food & Groceries", "Food Delivery"),
    ("order sa pagkain", "Food & Groceries", "Food Delivery"),
    ("order sa delivery ng pagkain", "Food & Groceries", "Food Delivery"),



    # Transportation entries
    ("Grab car to work", "Transportation", "Public Transit"),
    ("Angkas ride", "Transportation", "Public Transit"),
    ("JoyRide booking", "Transportation", "Public Transit"),
    ("Beep card load", "Transportation", "Public Transit"),
    ("MRT fare", "Transportation", "Public Transit"),
    ("LRT ticket", "Transportation", "Public Transit"),
    ("P2P bus fare", "Transportation", "Public Transit"),
    ("UV Express fare", "Transportation", "Public Transit"),
    ("jeepney fare", "Transportation", "Public Transit"),
    ("tricycle fare", "Transportation", "Public Transit"),
    ("ride hailing service", "Transportation", "Public Transit"),
    ("bus fare", "Transportation", "Public Transit"),
    ("train ticket", "Transportation", "Public Transit"),
    ("taxi fare", "Transportation", "Public Transit"),
    ("motorcycle taxi", "Transportation", "Public Transit"),
    ("pamasahe", "Transportation", "Public Transit"),
    ("sa bus", "Transportation", "Public Transit"),
    ("sa jeep", "Transportation", "Public Transit"),
    ("sa trike", "Transportation", "Public Transit"),
    ("transpo", "Transportation", "Public Transit"),

    ("Shell gas station", "Transportation", "Fuel/Gas"),
    ("Petron fuel", "Transportation", "Fuel/Gas"),
    ("Caltex gas", "Transportation", "Fuel/Gas"),
    ("fuel purchase", "Transportation", "Fuel/Gas"),
    ("gasoline refill", "Transportation", "Fuel/Gas"),
    ("diesel fuel", "Transportation", "Fuel/Gas"),
    ("gasolina", "Transportation", "Fuel/Gas"),
    ("krudo", "Transportation", "Fuel/Gas"),
    ("diesel", "Transportation", "Fuel/Gas"),
    ("unleaded", "Transportation", "Fuel/Gas"),
    ("unleaded gas", "Transportation", "Fuel/Gas"),

    ("Car change oil", "Transportation", "Vehicle Maintenance"),
    ("Tire replacement", "Transportation", "Vehicle Maintenance"),
    ("Car wash", "Transportation", "Vehicle Maintenance"),
    ("parking fee", "Transportation", "Vehicle Maintenance"),
    ("car registration", "Transportation", "Vehicle Maintenance"),
    ("LTO renewal", "Transportation", "Vehicle Maintenance"),
    ("car battery", "Transportation", "Vehicle Maintenance"),
    ("vehicle maintenance", "Transportation", "Vehicle Maintenance"),
    ("car service", "Transportation", "Vehicle Maintenance"),
    ("vehicle repair", "Transportation", "Vehicle Maintenance"),
    ("change oil", "Transportation", "Vehicle Maintenance"),
    ("talyer", "Transportation", "Vehicle Maintenance"),
    ("mekaniko", "Transportation", "Vehicle Maintenance"),
    ("parking", "Transportation", "Vehicle Maintenance"),
    ("renewal ng sasakyan", "Transportation", "Vehicle Maintenance"),

    ("Car insurance monthly", "Insurance", "Car Insurance"),
    ("car insurance", "Insurance", "Car Insurance"),
    ("auto insurance", "Insurance", "Car Insurance"),
    


     # Home Maintenance (expanded)
    ("Hardware store supplies", "Home Maintenance", "Home Improvements"),
    ("Paint and painting supplies", "Home Maintenance", "Home Improvements"),
    ("Home depot purchase", "Home Maintenance", "Home Improvements"),
    ("Home renovation material", "Home Maintenance", "Home Improvements"),
    ("Garden maintenance tools", "Home Maintenance", "Home Improvements"),
    ("Home security system", "Home Maintenance", "Home Improvements"),
    ("painting supplies", "Home Maintenance", "Home Improvements"),
    ("home renovation", "Home Maintenance", "Home Improvements"),
    ("garden tools", "Home Maintenance", "Home Improvements"),
    ("home security", "Home Maintenance", "Home Improvements"),
    ("home improvement", "Home Maintenance", "Home Improvements"),
    ("home maintenance", "Home Maintenance", "Home Improvements"),
    ("house renovation", "Home Maintenance", "Home Improvements"),
    ("house painting", "Home Maintenance", "Home Improvements"),
    ("house maintenance", "Home Maintenance", "Home Improvements"),
    ("bahay at lupa", "Home Maintenance", "Home Improvements"),
    ("kagamitan sa bahay", "Home Maintenance", "Home Improvements"),
    ("pintura", "Home Maintenance", "Home Improvements"),
    ("kagamitan sa hardin", "Home Maintenance", "Home Improvements"),
    ("kagamitan sa kusina", "Home Maintenance", "Home Improvements"),
    ("kagamitan sa banyo", "Home Maintenance", "Home Improvements"),
    ("kagamitan sa sala", "Home Maintenance", "Home Improvements"),
    ("kagamitan sa kwarto", "Home Maintenance", "Home Improvements"),
    ("kagamitan sa labas", "Home Maintenance", "Home Improvements"),
    ("kagamitan sa loob", "Home Maintenance", "Home Improvements"),
    ("kagamitan sa labas", "Home Maintenance", "Home Improvements"),
    ("kagamitan sa bahay", "Home Maintenance", "Home Improvements"),
    ("kagamitan sa kusina", "Home Maintenance", "Home Improvements"),
    ("kagamitan sa banyo", "Home Maintenance", "Home Improvements"),
    ("kagamitan sa sala", "Home Maintenance", "Home Improvements"),

    ("Electrical repair materials", "Home Maintenance", "Repairs"),
    ("Plumbing repair kit", "Home Maintenance", "Repairs"),
    ("Furniture repair kit", "Home Maintenance", "Repairs"),
    ("Aircon cleaning service", "Home Maintenance", "Repairs"),
    ("Pest control service", "Home Maintenance", "Repairs"),
    ("Window repair", "Home Maintenance", "Repairs"),
    ("Roof repair materials", "Home Maintenance", "Repairs"),
    ("kabayo ng bahay", "Home Maintenance", "Repairs"),
    ("gamit sa pagkumpuni", "Home Maintenance", "Repairs"),
    ("repairs sa bahay", "Home Maintenance", "Repairs"),
    ("furniture repair", "Home Maintenance", "Repairs"),
    ("pest control", "Home Maintenance", "Repairs"),
    ("window replacement", "Home Maintenance", "Repairs"),
    ("roof repair", "Home Maintenance", "Repairs"),
    ("house repair", "Home Maintenance", "Repairs"),
    ("pagkukumpuni", "Home Maintenance", "Repairs"),
    ("Floor maintenance supplies", "Home Maintenance", "Cleaning Supplies"),
    ("floor cleaning", "Home Maintenance", "Cleaning Supplies"),
    ("house cleaning", "Home Maintenance", "Cleaning Supplies"),
    ("cleaning materials", "Home Maintenance", "Cleaning Supplies"),
    ("cleaning supplies", "Home Maintenance", "Cleaning Supplies"),
    ("cleaning tools", "Home Maintenance", "Cleaning Supplies"),
    ("cleaning equipment", "Home Maintenance", "Cleaning Supplies"),
    ("cleaning products", "Home Maintenance", "Cleaning Supplies"),
    ("cleaning chemicals", "Home Maintenance", "Cleaning Supplies"),
    ("cleaning agents", "Home Maintenance", "Cleaning Supplies"),
    ("cleaning solutions", "Home Maintenance", "Cleaning Supplies"),
    ("walis tambo", "Home Maintenance", "Cleaning Supplies"),
    ("walis tingting", "Home Maintenance", "Cleaning Supplies"),
    ("basahan", "Home Maintenance", "Cleaning Supplies"),


    # Healthcare entries
    ("Mercury Drug medicines", "Healthcare", "Medications"),
    ("Watsons pharmacy", "Healthcare", "Medications"),
    ("South Star Drug purchase", "Healthcare", "Medications"),
    ("The Generics Pharmacy", "Healthcare", "Medications"),
    ("Rose Pharmacy", "Healthcare", "Medications"),
    ("medicine purchase", "Healthcare", "Medications"),
    ("prescription drugs", "Healthcare", "Medications"),
    ("over the counter medicine", "Healthcare", "Medications"),
    ("gamot", "Healthcare", "Medications"),
    ("medisina", "Healthcare", "Medications"),
    ("reseta", "Healthcare", "Medications"),

    ("Medical checkup St. Luke's", "Healthcare", "Doctor Visits"),
    ("flu vaccine", "Healthcare", "Doctor Visits"),
    ("annual physical", "Healthcare", "Doctor Visits"),
    ("laboratory test", "Healthcare", "Doctor Visits"),
    ("blood test", "Healthcare", "Doctor Visits"),
    ("X-ray", "Healthcare", "Doctor Visits"),
    ("ultrasound", "Healthcare", "Doctor Visits"),
    ("doctor consultation", "Healthcare", "Doctor Visits"),
    ("medical checkup", "Healthcare", "Doctor Visits"),
    ("check-up sa doktor", "Healthcare", "Doctor Visits"),
    ("pagpapatingin", "Healthcare", "Doctor Visits"),
    ("konsulta", "Healthcare", "Doctor Visits"),

    ("Dental cleaning", "Healthcare", "Dental Care"),
    ("Braces adjustment", "Healthcare", "Dental Care"),
    ("pasta ng ngipin", "Healthcare", "Dental Care"),
    ("teeth cleaning", "Healthcare", "Dental Care"),
    ("tooth extraction", "Healthcare", "Dental Care"),
    ("pasta sa ngipin", "Healthcare", "Dental Care"),
    ("Root canal", "Healthcare", "Dental Care"),
    ("dental checkup", "Healthcare", "Dental Care"),
    ("tooth filling", "Healthcare", "Dental Care"),
    ("pasta sa ngipin", "Healthcare", "Dental Care"),
    ("bunot ngipin", "Healthcare", "Dental Care"),
    ("linis ngipin", "Healthcare", "Dental Care"),

    ("Eye checkup", "Healthcare", "Vision Care/Eyewear"),
    ("EO glasses", "Healthcare", "Vision Care/Eyewear"),
    ("Ideal Vision contact lenses", "Healthcare", "Vision Care/Eyewear"),
    ("eye examination", "Healthcare", "Vision Care/Eyewear"),
    ("contact lenses", "Healthcare", "Vision Care/Eyewear"),
    ("eyeglasses", "Healthcare", "Vision Care/Eyewear"),
    ("salamin sa mata", "Healthcare", "Vision Care/Eyewear"),
    ("contact lens", "Healthcare", "Vision Care/Eyewear"),
    ("check-up mata", "Healthcare", "Vision Care/Eyewear"),
    ("salamin", "Healthcare", "Vision Care/Eyewear"),
    ("lens", "Healthcare", "Vision Care/Eyewear"),

    ("Therapy session", "Healthcare", "Mental Health Services"),
    ("Psychologist consultation", "Healthcare", "Mental Health Services")
    ("counseling session", "Healthcare", "Mental Health Services"),
    ("mental health consultation", "Healthcare", "Mental Health Services")
    ("therapy", "Healthcare", "Mental Health Services"),
    ("psychologist", "Healthcare", "Mental Health Services"),
    ("counseling", "Healthcare", "Mental Health Services"),
    ("mental health", "Healthcare", "Mental Health Services"),

    # Entertainment
    ("SM Cinema tickets", "Entertainment", "Movies/Shows"),
    ("Ayala Cinema movie", "Entertainment", "Movies/Shows"),
    ("movie ticket", "Entertainment", "Movies/Shows"),
    ("cinema admission", "Entertainment", "Movies/Shows"),
    ("sine", "Entertainment", "Movies/Shows"),
    ("pelikula", "Entertainment", "Movies/Shows"),

    ("concert ticket", "Entertainment", "Concert Tickets/Events"),
    ("event admission", "Entertainment", "Concert Tickets/Events"),
    ("MOA concert tickets", "Entertainment", "Concert Tickets/Events"),
    ("Araneta show tickets", "Entertainment", "Concert Tickets/Events"),
    ("concert ticket", "Entertainment", "Concert Tickets/Events"),
    ("event admission", "Entertainment", "Concert Tickets/Events"),
    ("concert", "Entertainment", "Concert Tickets/Events"),
    ("pustiso", "Entertainment", "Concert Tickets/Events"),
    ("Philippine Aren Concert", "Entertainment", "Concert Tickets/Events"),
    ("Araneta Coliseum concert", "Entertainment", "Concert Tickets/Events"),
    ("K-Pop concert", "Entertainment", "Concert Tickets/Events"),
    ("K-Pop event", "Entertainment", "Concert Tickets/Events"),
    ("P-Pop concert", "Entertainment", "Concert Tickets/Events"),
    ("P-Pop event", "Entertainment", "Concert Tickets/Events"),
    ("International Artist", "Entertainment", "Concert Tickets/Events"),
    ("International Concert", "Entertainment", "Concert Tickets/Events"),
    ("International Event", "Entertainment", "Concert Tickets/Events"),
    ("Local Artist", "Entertainment", "Concert Tickets/Events"),
    ("Local Concert", "Entertainment", "Concert Tickets/Events"),
    ("Local Event", "Entertainment", "Concert Tickets/Events"),
    ("Local Band", "Entertainment", "Concert Tickets/Events"),
    ("Local Singer", "Entertainment", "Concert Tickets/Events"),
    
    ("Steam games purchase", "Entertainment", "Gaming"),
    ("Nintendo Switch game", "Entertainment", "Gaming"),
    ("PlayStation Plus subscription", "Entertainment", "Gaming"),
    ("Xbox Game Pass", "Entertainment", "Gaming"),
    ("video game", "Entertainment", "Gaming"),
    ("game subscription", "Entertainment", "Gaming"),
    ("video games", "Entertainment", "Gaming"),
    ("laro", "Entertainment", "Gaming"),
    
    ("libangan", "Entertainment", "Hobbies"),
    ("hobby", "Entertainment", "Hobbies"),
    ("art materials", "Entertainment", "Hobbies"),
    ("art supplies", "Entertainment", "Hobbies"),
    ("craft materials", "Entertainment", "Hobbies"),
    ("BGC art supplies", "Entertainment", "Hobbies"),
    ("National Bookstore art materials", "Entertainment", "Hobbies"),
    ("hobby materials", "Entertainment", "Hobbies"),
    ("craft supplies", "Entertainment", "Hobbies"),



     # Education
    ("National Bookstore supplies", "Education", "Supplies"),
    ("School tuition fee", "Education", "Tuition"),
    ("Online course Udemy", "Education", "Courses"),
    ("Coursera subscription", "Education", "Courses"),
    ("LinkedIn Learning", "Education", "Courses"),
    ("School books", "Education", "Books"),
    ("Review center fee", "Education", "Tuition"),
    ("study materials", "Education", "Supplies"),
    ("school supplies", "Education", "Supplies"),
    ("semester payment", "Education", "Tuition"),
    ("tuition fee", "Education", "Tuition"),
    ("online learning", "Education", "Courses"),
    ("skill development course", "Education", "Courses"),
    ("textbooks", "Education", "Books"),
    ("reference materials", "Education", "Books"),
    ("bayad sa eskwela", "Education", "Tuition"),
    ("matrikula", "Education", "Tuition"),
    ("aral", "Education", "Tuition"),
    ("libro", "Education", "Books"),
    ("notebooks", "Education", "Supplies"),
    ("papel", "Education", "Supplies"),
    ("online class", "Education", "Courses"),
    ("seminar", "Education", "Courses"),
    ("workshop", "Education", "Courses"),
    ("review center", "Education", "Tuition"),
    ("review materials", "Education", "Supplies"),
    ("school project", "Education", "Supplies"),
    ("school activity", "Education", "Supplies"),
    ("school event", "Education", "Supplies"),
    ("school requirement", "Education", "Supplies"),
    ("school uniform", "Education", "Supplies"),
    ("school supplies", "Education", "Supplies"),
    ("school payment", "Education", "Tuition"),
    ("school fee", "Education", "Tuition"),
    ("online course", "Education", "Courses"),
    ("online learning", "Education", "Courses"),
    ("online training", "Education", "Courses"),
    ("online seminar", "Education", "Courses"),
    ("online workshop", "Education", "Courses"),
    ("online review", "Education", "Courses"),
    ("online class", "Education", "Courses"),
    ("online education", "Education", "Courses"),
    ("online certification", "Education", "Courses"),
    ("online degree", "Education", "Courses"),
    ("online diploma", "Education", "Courses"),
    ("online training", "Education", "Courses"),
    ("online seminar", "Education", "Courses"),
    ("online workshop", "Education", "Courses"),
    ("online review", "Education", "Courses"),
    ("online class", "Education", "Courses"),
    ("online education", "Education", "Courses"),
    ("online certification", "Education", "Courses"),
    ("online degree", "Education", "Courses"),
    ("online diploma", "Education", "Courses"),
    ("online training", "Education", "Courses"),
    ("online seminar", "Education", "Courses"),
    ("online workshop", "Education", "Courses"),
    ("online review", "Education", "Courses"),
    ("online class", "Education", "Courses"),
    ("online education", "Education", "Courses"),
    ("online certification", "Education", "Courses"),
    ("online degree", "Education", "Courses"),
    ("online diploma", "Education", "Courses"),
    ("online training", "Education", "Courses"),
    ("online seminar", "Education", "Courses"),
    ("online workshop", "Education", "Courses"),
    ("online review", "Education", "Courses"),
    ("online class", "Education", "Courses"),

    # Personal Care entries
    ("Watsons personal items", "Personal Care", "Personal Hygiene Products"),
    ("Guardian toiletries", "Personal Care", "Personal Hygiene Products"),
    ("Bench grooming items", "Personal Care", "Personal Hygiene Products"),
    ("Salon haircut", "Personal Care", "Haircuts/Salon Services"),
    ("Hair color treatment", "Personal Care", "Haircuts/Salon Services"),
    ("Facial treatment", "Personal Care", "Skincare Products"),
    ("Sephora skincare", "Personal Care", "Skincare Products"),
    ("Anytime Fitness membership", "Personal Care", "Gym/Fitness"),
    ("Gold's Gym monthly", "Personal Care", "Gym/Fitness"),
    ("Yoga class", "Personal Care", "Gym/Fitness"),
    ("shampoo and soap", "Personal Care", "Personal Hygiene Products"),
    ("deodorant", "Personal Care", "Personal Hygiene Products"),
    ("toothpaste", "Personal Care", "Personal Hygiene Products"),
    ("face mask", "Personal Care", "Skincare Products"),
    ("toner", "Personal Care", "Skincare Products"),
    ("moisturizer", "Personal Care", "Skincare Products"),
    ("makeup", "Personal Care", "Personal Hygiene Products"),
    ("nail polish", "Personal Care", "Personal Hygiene Products"),
    ("manicure pedicure", "Personal Care", "Haircuts/Salon Services"),
    ("gym trainer", "Personal Care", "Gym/Fitness"),
    ("boxing class", "Personal Care", "Gym/Fitness"),
    ("Zumba class", "Personal Care", "Gym/Fitness"),
    ("toiletries", "Personal Care", "Personal Hygiene Products"),
    ("bathroom supplies", "Personal Care", "Personal Hygiene Products"),
    ("hair products", "Personal Care", "Personal Hygiene Products"),
    ("haircut service", "Personal Care", "Haircuts/Salon Services"),
    ("hair treatment", "Personal Care", "Haircuts/Salon Services"),
    ("skin care products", "Personal Care", "Skincare Products"),
    ("facial products", "Personal Care", "Skincare Products"),
    ("gym membership", "Personal Care", "Gym/Fitness"),
    ("fitness class", "Personal Care", "Gym/Fitness"),
    ("exercise equipment", "Personal Care", "Gym/Fitness"),
     ("shampoo at sabon", "Personal Care", "Personal Hygiene Products"),
    ("deodorant", "Personal Care", "Personal Hygiene Products"),
    ("toothbrush", "Personal Care", "Personal Hygiene Products"),
    ("gupit", "Personal Care", "Haircuts/Salon Services"),
    ("salon", "Personal Care", "Haircuts/Salon Services"),
    ("rebond", "Personal Care", "Haircuts/Salon Services"),
    ("pampaganda", "Personal Care", "Skincare Products"),
    ("facial", "Personal Care", "Skincare Products"),
    ("gym", "Personal Care", "Gym/Fitness"),
    ("zumba", "Personal Care", "Gym/Fitness"),
    ("exercise", "Personal Care", "Gym/Fitness"),
    ("Men's grooming kit", "Personal Care", "Personal Hygiene Products"),
    ("Beard trimmer", "Personal Care", "Personal Hygiene Products"),
    ("Sun protection cream", "Personal Care", "Skincare Products"),
    ("Protein supplements", "Personal Care", "Gym/Fitness"),
    ("Fitness tracker", "Personal Care", "Gym/Fitness"),
    ("Group fitness class", "Personal Care", "Gym/Fitness"),
    ("Sports massage", "Personal Care", "Haircuts/Salon Services"),
    ("Wellness workshop", "Personal Care", "Gym/Fitness"),
    ("pilates class", "Personal Care", "Gym/Fitness"),
    ("supplement", "Personal Care", "Gym/Fitness"),
    ("fitness equipment", "Personal Care", "Gym/Fitness"),

    # Family/Dependents entries
    ("Pet food Pedigree", "Family/Dependents", "Pet Care/Supplies"),
    ("Vet checkup", "Family/Dependents", "Pet Care/Supplies"),
    ("Dog grooming", "Family/Dependents", "Pet Care/Supplies"),
    ("School supplies for kids", "Family/Dependents", "School Expenses"),
    ("Daycare monthly", "Family/Dependents", "Childcare"),
    ("yayas salary", "Family/Dependents", "Childcare"),
    ("yaya advance", "Family/Dependents", "Childcare"),
    ("dog food", "Family/Dependents", "Pet Care/Supplies"),
    ("cat food", "Family/Dependents", "Pet Care/Supplies"),
    ("pet vitamins", "Family/Dependents", "Pet Care/Supplies"),
    ("school project", "Family/Dependents", "School Expenses"),
    ("school uniform", "Family/Dependents", "School Expenses"),
    ("field trip fee", "Family/Dependents", "School Expenses"),
    ("school event", "Family/Dependents", "School Expenses"),
    ("pet supplies", "Family/Dependents", "Pet Care/Supplies"),
    ("animal food", "Family/Dependents", "Pet Care/Supplies"),
    ("veterinary service", "Family/Dependents", "Pet Care/Supplies"),
    ("school materials", "Family/Dependents", "School Expenses"),
    ("educational supplies", "Family/Dependents", "School Expenses"),
    ("childcare service", "Family/Dependents", "Childcare"),
    ("babysitting fee", "Family/Dependents", "Childcare"),
    ("nanny payment", "Family/Dependents", "Childcare"),
    ("pagkain ng alagang hayop", "Family/Dependents", "Pet Care/Supplies"),
    ("pagkain ng pusa", "Family/Dependents", "Pet Care/Supplies"),
    ("pagkain ng aso", "Family/Dependents", "Pet Care/Supplies"),
    ("bayad sa yaya", "Family/Dependents", "Childcare"),
    ("sweldo ng yaya", "Family/Dependents", "Childcare"),
    ("school supplies", "Family/Dependents", "School Expenses"),
    ("uniporme", "Family/Dependents", "School Expenses"),
    ("baon", "Family/Dependents", "School Expenses"),
    ("project sa school", "Family/Dependents", "School Expenses"),


    # Insurance entries
    ("Philhealth contribution", "Insurance", "Health Insurance"),
    ("HMO premium", "Insurance", "Health Insurance"),
    ("SSS contribution", "Insurance", "Health Insurance"),
    ("Sun Life insurance", "Insurance", "Life Insurance"),
    ("AXA insurance premium", "Insurance", "Life Insurance"),
    ("Malayan insurance", "Insurance", "Home Insurance"),
    ("Travel insurance for vacation", "Insurance", "Travel Insurance"),
    ("Pru Life payment", "Insurance", "Life Insurance"),
    ("Pru Life premium", "Insurance", "Life Insurance"),
    ("Manulife insurance", "Insurance", "Life Insurance"),
    ("BPI-AIA insurance", "Insurance", "Life Insurance"),
    ("car insurance annual", "Insurance", "Car Insurance"),
    ("SSS voluntary", "Insurance", "Health Insurance"),
    ("Maxicare premium", "Insurance", "Health Insurance"),
    ("Medicard payment", "Insurance", "Health Insurance"),
    ("insurance policy", "Insurance", "Life Insurance"),
    ("insurance premium", "Insurance", "Life Insurance"),
    ("insurance contribution", "Insurance", "Life Insurance"),
    ("insurance payment", "Insurance", "Life Insurance"),
    ("insurance fee", "Insurance", "Life Insurance"),
    ("insurance coverage", "Insurance", "Life Insurance"),
    ("insurance claim", "Insurance", "Life Insurance"),
    ("insurance policy", "Insurance", "Life Insurance"),
    ("insurance premium", "Insurance", "Life Insurance"),
    ("insurance contribution", "Insurance", "Life Insurance"),
    



     # Savings & Investments (expanded significantly)
    ("Bank of the Philippine Islands deposit", "Savings & Investments", "Retirement"),
    ("Metrobank time deposit", "Savings & Investments", "Investments"),
    ("Security Bank investment", "Savings & Investments", "Investments"),
    ("UITF investment", "Savings & Investments", "Investments"),
    ("Stock market purchase", "Savings & Investments", "Investments"),
    ("Mutual fund contribution", "Savings & Investments", "Investments"),
    ("Emergency fund deposit", "Savings & Investments", "Emergency Fund"),
    ("Retirement account contribution", "Savings & Investments", "Retirement"),
    ("Personal savings transfer", "Savings & Investments", "Savings Goals"),
    ("COL Financial stock buy", "Savings & Investments", "Investments"),
    ("First Metro Securities trading", "Savings & Investments", "Investments"),
    ("SSS voluntary contribution", "Savings & Investments", "Retirement"),
    ("PERA account deposit", "Savings & Investments", "Retirement"),
    ("Bonds investment", "Savings & Investments", "Investments"),
    ("Cryptocurrency investment", "Savings & Investments", "Investments"),
    ("Money market fund", "Savings & Investments", "Investments"),
    ("Investment trust fund", "Savings & Investments", "Investments"),
    ("Pagibig MP2 savings", "Savings & Investments", "Savings Goals"),
    ("iSave account deposit", "Savings & Investments", "Savings Goals"),
    ("alkansya deposit", "Savings & Investments", "Savings Goals"),
    ("ipon sa bangko", "Savings & Investments", "Savings Goals"),
    ("pamuhunan", "Savings & Investments", "Investments"),
    ("retirement fund", "Savings & Investments", "Retirement"),
    ("emergency savings", "Savings & Investments", "Emergency Fund"),
    ("savings account", "Savings & Investments", "Savings Goals"),
    ("personal savings", "Savings & Investments", "Savings Goals"),
    ("stock market", "Savings & Investments", "Investments"),
    ("mutual fund", "Savings & Investments", "Investments"),
    ("bonds", "Savings & Investments", "Investments"),
    ("cryptocurrency", "Savings & Investments", "Investments"),
    ("money market", "Savings & Investments", "Investments"),
    ("investment trust", "Savings & Investments", "Investments"),
    ("pag-ibig savings", "Savings & Investments", "Savings Goals"),
    ("alkansya", "Savings & Investments", "Savings Goals"),
    ("ipon", "Savings & Investments", "Savings Goals"),
    ("investment", "Savings & Investments", "Investments"),
    ("investment account", "Savings & Investments", "Investments"),
    ("investment portfolio", "Savings & Investments", "Investments"),
    ("investment strategy", "Savings & Investments", "Investments"),
    ("investment vehicle", "Savings & Investments", "Investments"),
    ("investment opportunity", "Savings & Investments", "Investments"),
    ("investment return", "Savings & Investments", "Investments"),
    ("investment risk", "Savings & Investments", "Investments"),
    ("investment performance", "Savings & Investments", "Investments"),
    ("investment growth", "Savings & Investments", "Investments"),
    ("investment loss", "Savings & Investments", "Investments"),
    ("investment gain", "Savings & Investments", "Investments"),
    ("investment income", "Savings & Investments", "Investments"),
    ("investment capital", "Savings & Investments", "Investments"),
    ("investment fund", "Savings & Investments", "Investments"),
    ("investment account", "Savings & Investments", "Investments"),
    ("investment portfolio", "Savings & Investments", "Investments"),
    ("investment strategy", "Savings & Investments", "Investments"),
    ("investment vehicle", "Savings & Investments", "Investments"),
    ("investment opportunity", "Savings & Investments", "Investments"),
    ("investment return", "Savings & Investments", "Investments"),
    
    # Shopping entries
    ("Uniqlo clothes", "Shopping", "Clothing"),
    ("H&M purchase", "Shopping", "Clothing"),
    ("Zara shopping", "Shopping", "Clothing"),
    ("SM Store clothes", "Shopping", "Clothing"),
    ("Landmark department store", "Shopping", "Clothing"),
    ("SM Appliance center", "Shopping", "Electronics"),
    ("Abenson appliance", "Shopping", "Electronics"),
    ("Lazada order", "Shopping", "Electronics"),
    ("Shopee purchase", "Shopping", "Electronics"),
    ("Mandaue Foam furniture", "Shopping", "Home Decor"),
    ("SM Home items", "Shopping", "Home Decor"),
    ("Christmas gift shopping", "Shopping", "Gifts"),
    ("Birthday present", "Shopping", "Gifts"),
    ("Divisoria shopping", "Shopping", "Clothing"),
    ("Taytay tiangge", "Shopping", "Clothing"),
    ("Greenhills shopping", "Shopping", "Electronics"),
    ("cellphone load", "Bills & Utilities", "Phone Plan"),
    ("birthday regalo", "Shopping", "Gifts"),
    ("Christmas exchange gift", "Shopping", "Gifts"),
    ("clothing purchase", "Shopping", "Clothing"),
    ("shoes shopping", "Shopping", "Clothing"),
    ("accessories", "Shopping", "Clothing"),
    ("electronic device", "Shopping", "Electronics"),
    ("gadget purchase", "Shopping", "Electronics"),
    ("home furniture", "Shopping", "Home Decor"),
    ("household items", "Shopping", "Home Decor"),
    ("gift purchase", "Shopping", "Gifts"),
    ("present shopping", "Shopping", "Gifts"),
    ("damit", "Shopping", "Clothing"),
    ("sapatos", "Shopping", "Clothing"),
    ("tsinelas", "Shopping", "Clothing"),
    ("gadyet", "Shopping", "Electronics"),
    ("appliance", "Shopping", "Electronics"),
    ("kagamitan", "Shopping", "Electronics"),
    ("muwebles", "Shopping", "Home Decor"),
    ("gamit bahay", "Shopping", "Home Decor"),
    ("regalo", "Shopping", "Gifts"),
    ("pasalubong", "Shopping", "Gifts"),
    ("aguinaldo", "Shopping", "Gifts"),

    # Contingency Fund entries
    ("emergency home repair", "Contingency Fund", "Emergency Repairs"),
    ("unexpected plumbing issue", "Contingency Fund", "Emergency Repairs"),
    ("roof leak repair", "Contingency Fund", "Emergency Repairs"),
    ("electrical system emergency", "Contingency Fund", "Emergency Repairs"),
    ("broken pipe fix", "Contingency Fund", "Emergency Repairs"),
    ("urgent home maintenance", "Contingency Fund", "Emergency Repairs"),
    ("emergency electrician", "Contingency Fund", "Emergency Repairs"),
    ("urgent aircon repair", "Contingency Fund", "Emergency Repairs"),
    ("immediate house repair", "Contingency Fund", "Emergency Repairs"),
    ("tubero emergency", "Contingency Fund", "Emergency Repairs"),
    ("biglang sira bubong", "Contingency Fund", "Emergency Repairs"),
    ("emergency repair bahay", "Contingency Fund", "Emergency Repairs"),
    ("emergency room visit", "Contingency Fund", "Medical Emergencies"),
    ("urgent care clinic", "Contingency Fund", "Medical Emergencies"),
    ("emergency medicine purchase", "Contingency Fund", "Medical Emergencies"),
    ("ambulance service", "Contingency Fund", "Medical Emergencies"),
    ("sudden illness treatment", "Contingency Fund", "Medical Emergencies"),
    ("unexpected medical procedure", "Contingency Fund", "Medical Emergencies"),
    ("emergency dental work", "Contingency Fund", "Medical Emergencies"),
    ("accident medical expense", "Contingency Fund", "Medical Emergencies"),
    ("emergency hospital admission", "Contingency Fund", "Medical Emergencies"),
    ("biglang karamdaman", "Contingency Fund", "Medical Emergencies"),
    ("emergency sa ospital", "Contingency Fund", "Medical Emergencies"),
    ("aksidente medical expense", "Contingency Fund", "Medical Emergencies"),
    ("unexpected car breakdown", "Contingency Fund", "Unexpected Expenses"),
    ("stolen wallet replacement", "Contingency Fund", "Unexpected Expenses"),
    ("sudden travel expense", "Contingency Fund", "Unexpected Expenses"),
    ("emergency accommodation", "Contingency Fund", "Unexpected Expenses"),
    ("unexpected tax payment", "Contingency Fund", "Unexpected Expenses"),
    ("sudden family emergency", "Contingency Fund", "Unexpected Expenses"),
    ("unplanned legal expense", "Contingency Fund", "Unexpected Expenses"),
    ("emergency fund withdrawal", "Contingency Fund", "Unexpected Expenses"),
    ("urgent document processing", "Contingency Fund", "Unexpected Expenses"),
    ("unexpected penalty fee", "Contingency Fund", "Unexpected Expenses"),
    ("biglang gastos", "Contingency Fund", "Unexpected Expenses"),
    ("hindi inaasahang bayarin", "Contingency Fund", "Unexpected Expenses"),
    ("emergency na gastusin", "Contingency Fund", "Unexpected Expenses"),
    ("di-inaasahang gastos", "Contingency Fund", "Unexpected Expenses")
    ("computer repair emergency", "Contingency Fund", "Unexpected Expenses"),
    ("smartphone replacement", "Contingency Fund", "Unexpected Expenses"),
    ("laptop screen repair", "Contingency Fund", "Unexpected Expenses"),
    ("urgent appliance replacement", "Contingency Fund", "Unexpected Expenses"),
    ("unexpected travel cancellation", "Contingency Fund", "Unexpected Expenses"),
    ("emergency visa processing", "Contingency Fund", "Unexpected Expenses"),
    ("biglang pagkakasakit", "Contingency Fund", "Medical Emergencies"),
    ("unexpected gadget repair", "Contingency Fund", "Unexpected Expenses"),
    ("emergency passport renewal", "Contingency Fund", "Unexpected Expenses")
]
class ExpenseClassifier:
    def __init__(self, training_data):
        """Initialize the classifier with training data"""
        try:
            self.training_data = training_data
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 3))

            # Prepare training texts and labels
            self.texts = [entry[0] for entry in training_data]
            self.categories = [entry[1] for entry in training_data]
            self.expense_types = [entry[2] for entry in training_data]

            # Fit vectorizer
            self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)
            print("Classifier initialized successfully")
        except Exception as e:
            print(f"Error initializing classifier: {e}")
            raise

    def preprocess_text(self, text):
        """Preprocess input text"""
        try:
            # Convert to lowercase
            text = text.lower()
            # Remove punctuation
            text = text.translate(str.maketrans('', '', string.punctuation))
            # Tokenize
            tokens = word_tokenize(text)
            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words]
            return ' '.join(tokens)
        except Exception as e:
            print(f"Error in preprocess_text: {e}")
            raise

    def classify_expense(self, user_input):
        """Classify user input into category and expense type"""
        try:
            # Preprocess input
            processed_input = self.preprocess_text(user_input)
            # Vectorize input
            input_vector = self.vectorizer.transform([processed_input])
            # Calculate similarities
            similarities = cosine_similarity(input_vector, self.tfidf_matrix).flatten()
            # Get best match
            best_match_idx = np.argmax(similarities)
            confidence_score = float(similarities[best_match_idx])
            
            return {
                'user_input': user_input,
                'category': self.categories[best_match_idx],
                'expense_type': self.expense_types[best_match_idx],
                'confidence_score': confidence_score,
                'requires_new_expense_type': confidence_score < 0.3
            }
        except Exception as e:
            print(f"Error in classify_expense: {e}")
            print(f"Input text: {user_input}")
            print(traceback.format_exc())
            raise

app = Flask(__name__)
CORS(app)

# Initialize classifier globally
try:
    classifier = ExpenseClassifier(training_data)
    print("Global classifier initialized")
except Exception as e:
    print(f"Error initializing global classifier: {e}")
    sys.exit(1)

@app.route('/classify_expense', methods=['POST'])
def classify_expense():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400
        
        expense_text = data.get('expense_text')
        if not expense_text:
            return jsonify({'error': 'No expense_text provided'}), 400

        print(f"Received request to classify: {expense_text}")
        
        result = classifier.classify_expense(expense_text)
        similar_entries = []  # We'll skip similar entries for now to simplify debugging
        
        response = {
            'classification': result,
            'similar_entries': similar_entries
        }
        
        print(f"Successful classification: {response}")
        return jsonify(response)

    except Exception as e:
        print(f"Error processing request: {e}")
        print(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
import numpy as np
import nltk
import string
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from flask import Flask, request, jsonify
from flask_cors import CORS


# Training data structure: (user_input, category, expense_type)
training_data = [

    # Bills & Utilities entries
    ("Meralco bill payment", "Bills & Utilities", "Electricity"),
    ("Maynilad water bill", "Bills & Utilities", "Water"),
    ("Manila Water payment", "Bills & Utilities", "Water"),
    ("PLDT Home Fiber monthly", "Bills & Utilities", "Internet"),
    ("Globe At Home internet", "Bills & Utilities", "Internet"),
    ("Converge ICT payment", "Bills & Utilities", "Internet"),
    ("Globe postpaid plan", "Bills & Utilities", "Phone Plan"),
    ("Smart postpaid bill", "Bills & Utilities", "Phone Plan"),
    ("Netflix subscription", "Bills & Utilities", "Streaming Services"),
    ("Spotify Premium", "Bills & Utilities", "Streaming Services"),
    ("Disney+ subscription", "Bills & Utilities", "Streaming Services"),
    ("Amazon Prime", "Bills & Utilities", "Streaming Services"),
    ("HBO Go monthly", "Bills & Utilities", "Streaming Services"),
    ("Viu Premium", "Bills & Utilities", "Streaming Services"),
    ("YouTube Premium", "Bills & Utilities", "Streaming Services"),
    ("Monthly rent BGC condo", "Bills & Utilities", "Rent/Mortgage"),
    ("House rental Quezon City", "Bills & Utilities", "Rent/Mortgage"),
    ("Condo association dues", "Bills & Utilities", "Rent/Mortgage"),
    ("Petron gas tank refill", "Bills & Utilities", "Gas"),
    ("Shell gas for stove", "Bills & Utilities", "Gas"),
    ("kuryente payment", "Bills & Utilities", "Electricity"),
    ("tubig bill", "Bills & Utilities", "Water"),
    ("rent payment apartment", "Bills & Utilities", "Rent/Mortgage"),
    ("house amortization", "Bills & Utilities", "Rent/Mortgage"),
    ("pag-ibig housing loan", "Bills & Utilities", "Rent/Mortgage"),
    ("prepaid load Globe", "Bills & Utilities", "Phone Plan"),
    ("Smart regular load", "Bills & Utilities", "Phone Plan"),
    ("TnT load", "Bills & Utilities", "Phone Plan"),
    ("monthly association dues", "Bills & Utilities", "Rent/Mortgage"),
    ("annual condo dues", "Bills & Utilities", "Rent/Mortgage"),
    ("Apple Music subscription", "Bills & Utilities", "Streaming Services"),
    ("WeTV subscription", "Bills & Utilities", "Streaming Services"),
    ("iQiyi monthly", "Bills & Utilities", "Streaming Services"),
    ("electricity bill", "Bills & Utilities", "Electricity"),
    ("power bill payment", "Bills & Utilities", "Electricity"),
    ("utility bill electric", "Bills & Utilities", "Electricity"),
    ("water bill payment", "Bills & Utilities", "Water"),
    ("monthly water utilities", "Bills & Utilities", "Water"),
    ("internet monthly bill", "Bills & Utilities", "Internet"),
    ("wifi service payment", "Bills & Utilities", "Internet"),
    ("broadband payment", "Bills & Utilities", "Internet"),
    ("mobile plan monthly", "Bills & Utilities", "Phone Plan"),
    ("phone bill payment", "Bills & Utilities", "Phone Plan"),
    ("streaming service monthly", "Bills & Utilities", "Streaming Services"),
    ("video streaming subscription", "Bills & Utilities", "Streaming Services"),
    ("music subscription", "Bills & Utilities", "Streaming Services"),
    ("monthly apartment rent", "Bills & Utilities", "Rent/Mortgage"),
    ("house payment", "Bills & Utilities", "Rent/Mortgage"),
    ("monthly mortgage", "Bills & Utilities", "Rent/Mortgage"),
    ("cooking gas refill", "Bills & Utilities", "Gas"),
    ("gas utility bill", "Bills & Utilities", "Gas"),
     ("bayad kuryente", "Bills & Utilities", "Electricity"),
    ("bayad ilaw", "Bills & Utilities", "Electricity"),
    ("kontador kuryente", "Bills & Utilities", "Electricity"),
    ("bayad tubig", "Bills & Utilities", "Water"),
    ("bill ng tubig", "Bills & Utilities", "Water"),
    ("bayad internet", "Bills & Utilities", "Internet"),
    ("wifi bill", "Bills & Utilities", "Internet"),
    ("load sa wifi", "Bills & Utilities", "Internet"),
    ("bayad telepono", "Bills & Utilities", "Phone Plan"),
    ("load sa phone", "Bills & Utilities", "Phone Plan"),
    ("renta bahay", "Bills & Utilities", "Rent/Mortgage"),
    ("upa ng apartment", "Bills & Utilities", "Rent/Mortgage"),
    ("bayad gas", "Bills & Utilities", "Gas"),
    ("tangke ng gas", "Bills & Utilities", "Gas"),
    ("gasul", "Bills & Utilities", "Gas"),

    # Food & Groceries entries
    ("SM Supermarket grocery", "Food & Groceries", "Groceries"),
    ("Puregold weekly groceries", "Food & Groceries", "Groceries"),
    ("Robinson's Supermarket", "Food & Groceries", "Groceries"),
    ("Marketplace by Rustan's", "Food & Groceries", "Groceries"),
    ("Landers shopping", "Food & Groceries", "Groceries"),
    ("S&R grocery shopping", "Food & Groceries", "Groceries"),
    ("Jollibee lunch", "Food & Groceries", "Dining Out"),
    ("McDonald's breakfast", "Food & Groceries", "Dining Out"),
    ("KFC dinner", "Food & Groceries", "Dining Out"),
    ("Chowking merienda", "Food & Groceries", "Dining Out"),
    ("Max's Restaurant", "Food & Groceries", "Dining Out"),
    ("Starbucks coffee", "Food & Groceries", "Coffee/Beverages"),
    ("CBTL drinks", "Food & Groceries", "Coffee/Beverages"),
    ("Tim Hortons coffee", "Food & Groceries", "Coffee/Beverages"),
    ("GrabFood delivery", "Food & Groceries", "Food Delivery"),
    ("Foodpanda order", "Food & Groceries", "Food Delivery"),
    ("Pick.A.Roo food delivery", "Food & Groceries", "Food Delivery"),
    ("MetroMart groceries delivery", "Food & Groceries", "Groceries"),
    ("palengke groceries", "Food & Groceries", "Groceries"),
    ("wet market", "Food & Groceries", "Groceries"),
    ("sari-sari store", "Food & Groceries", "Groceries"),
    ("7-Eleven snacks", "Food & Groceries", "Groceries"),
    ("Ministop food", "Food & Groceries", "Dining Out"),
    ("Uncle John's food", "Food & Groceries", "Dining Out"),
    ("Mang Inasal", "Food & Groceries", "Dining Out"),
    ("Goldilocks", "Food & Groceries", "Dining Out"),
    ("Greenwich pizza", "Food & Groceries", "Dining Out"),
    ("Yellow Cab delivery", "Food & Groceries", "Food Delivery"),
    ("Angel's Pizza", "Food & Groceries", "Food Delivery"),
    ("Dunkin Donuts", "Food & Groceries", "Coffee/Beverages"),
    ("Ministop coffee", "Food & Groceries", "Coffee/Beverages"),
    ("Uncle John's coffee", "Food & Groceries", "Coffee/Beverages"),
    ("supermarket shopping", "Food & Groceries", "Groceries"),
    ("weekly grocery run", "Food & Groceries", "Groceries"),
    ("food supplies", "Food & Groceries", "Groceries"),
    ("fresh produce", "Food & Groceries", "Groceries"),
    ("meat and poultry", "Food & Groceries", "Groceries"),
    ("breakfast meal", "Food & Groceries", "Dining Out"),
    ("lunch restaurant", "Food & Groceries", "Dining Out"),
    ("dinner out", "Food & Groceries", "Dining Out"),
    ("afternoon snack", "Food & Groceries", "Dining Out"),
    ("coffee shop", "Food & Groceries", "Coffee/Beverages"),
    ("milk tea", "Food & Groceries", "Coffee/Beverages"),
    ("bubble tea", "Food & Groceries", "Coffee/Beverages"),
    ("food delivery order", "Food & Groceries", "Food Delivery"),
    ("online food order", "Food & Groceries", "Food Delivery"),
    ("meal delivery", "Food & Groceries", "Food Delivery"),
    ("pamibili sa palengke", "Food & Groceries", "Groceries"),
    ("grocery sa supermarket", "Food & Groceries", "Groceries"),
    ("mga bigas", "Food & Groceries", "Groceries"),
    ("ulam", "Food & Groceries", "Groceries"),
    ("pagkain", "Food & Groceries", "Groceries"),
    ("gulay", "Food & Groceries", "Groceries"),
    ("karne", "Food & Groceries", "Groceries"),
    ("almusal sa labas", "Food & Groceries", "Dining Out"),
    ("tanghalian", "Food & Groceries", "Dining Out"),
    ("hapunan sa labas", "Food & Groceries", "Dining Out"),
    ("meryenda", "Food & Groceries", "Dining Out"),
    ("kainan sa labas", "Food & Groceries", "Dining Out"),
    ("delivery ng pagkain", "Food & Groceries", "Food Delivery"),
    ("padeliver na food", "Food & Groceries", "Food Delivery"),
    ("kape sa labas", "Food & Groceries", "Coffee/Beverages"),



    # Healthcare entries
    ("Mercury Drug medicines", "Healthcare", "Medications"),
    ("Watsons pharmacy", "Healthcare", "Medications"),
    ("South Star Drug purchase", "Healthcare", "Medications"),
    ("The Generics Pharmacy", "Healthcare", "Medications"),
    ("Medical checkup St. Luke's", "Healthcare", "Doctor Visits"),
    ("Dental cleaning", "Healthcare", "Dental Care"),
    ("Braces adjustment", "Healthcare", "Dental Care"),
    ("Eye checkup", "Healthcare", "Vision Care/Eyewear"),
    ("EO glasses", "Healthcare", "Vision Care/Eyewear"),
    ("Ideal Vision contact lenses", "Healthcare", "Vision Care/Eyewear"),
    ("Therapy session", "Healthcare", "Mental Health Services"),
    ("Psychologist consultation", "Healthcare", "Mental Health Services"),
    ("pasta ng ngipin", "Healthcare", "Dental Care"),
    ("teeth cleaning", "Healthcare", "Dental Care"),
    ("tooth extraction", "Healthcare", "Dental Care"),
    ("pasta sa ngipin", "Healthcare", "Dental Care"),
    ("Root canal", "Healthcare", "Dental Care"),
    ("flu vaccine", "Healthcare", "Doctor Visits"),
    ("annual physical", "Healthcare", "Doctor Visits"),
    ("laboratory test", "Healthcare", "Doctor Visits"),
    ("blood test", "Healthcare", "Doctor Visits"),
    ("X-ray", "Healthcare", "Doctor Visits"),
    ("ultrasound", "Healthcare", "Doctor Visits"),
    ("Rose Pharmacy", "Healthcare", "Medications"),
    ("medicine purchase", "Healthcare", "Medications"),
    ("prescription drugs", "Healthcare", "Medications"),
    ("over the counter medicine", "Healthcare", "Medications"),
    ("doctor consultation", "Healthcare", "Doctor Visits"),
    ("medical checkup", "Healthcare", "Doctor Visits"),
    ("dental checkup", "Healthcare", "Dental Care"),
    ("tooth filling", "Healthcare", "Dental Care"),
    ("eye examination", "Healthcare", "Vision Care/Eyewear"),
    ("contact lenses", "Healthcare", "Vision Care/Eyewear"),
    ("eyeglasses", "Healthcare", "Vision Care/Eyewear"),
    ("counseling session", "Healthcare", "Mental Health Services"),
    ("mental health consultation", "Healthcare", "Mental Health Services"),
    ("gamot", "Healthcare", "Medications"),
    ("medisina", "Healthcare", "Medications"),
    ("reseta", "Healthcare", "Medications"),
    ("check-up sa doktor", "Healthcare", "Doctor Visits"),
    ("pagpapatingin", "Healthcare", "Doctor Visits"),
    ("konsulta", "Healthcare", "Doctor Visits"),
    ("pasta sa ngipin", "Healthcare", "Dental Care"),
    ("bunot ngipin", "Healthcare", "Dental Care"),
    ("linis ngipin", "Healthcare", "Dental Care"),
    ("salamin sa mata", "Healthcare", "Vision Care/Eyewear"),
    ("contact lens", "Healthcare", "Vision Care/Eyewear"),
    ("check-up mata", "Healthcare", "Vision Care/Eyewear"),

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


    # Transportation entries
    ("Grab car to work", "Transportation", "Public Transit"),
    ("Angkas ride", "Transportation", "Public Transit"),
    ("JoyRide booking", "Transportation", "Public Transit"),
    ("Beep card load", "Transportation", "Public Transit"),
    ("MRT fare", "Transportation", "Public Transit"),
    ("LRT ticket", "Transportation", "Public Transit"),
    ("P2P bus fare", "Transportation", "Public Transit"),
    ("Shell gas station", "Transportation", "Fuel/Gas"),
    ("Petron fuel", "Transportation", "Fuel/Gas"),
    ("Caltex gas", "Transportation", "Fuel/Gas"),
    ("Car change oil", "Transportation", "Vehicle Maintenance"),
    ("Tire replacement", "Transportation", "Vehicle Maintenance"),
    ("Car wash", "Transportation", "Vehicle Maintenance"),
    ("Car insurance monthly", "Insurance", "Car Insurance"),
    ("UV Express fare", "Transportation", "Public Transit"),
    ("jeepney fare", "Transportation", "Public Transit"),
    ("tricycle fare", "Transportation", "Public Transit"),
    ("parking fee", "Transportation", "Vehicle Maintenance"),
    ("car registration", "Transportation", "Vehicle Maintenance"),
    ("LTO renewal", "Transportation", "Vehicle Maintenance"),
    ("car battery", "Transportation", "Vehicle Maintenance"),
    ("car insurance", "Insurance", "Car Insurance"),
    ("ride hailing service", "Transportation", "Public Transit"),
    ("bus fare", "Transportation", "Public Transit"),
    ("train ticket", "Transportation", "Public Transit"),
    ("taxi fare", "Transportation", "Public Transit"),
    ("motorcycle taxi", "Transportation", "Public Transit"),
    ("fuel purchase", "Transportation", "Fuel/Gas"),
    ("gasoline refill", "Transportation", "Fuel/Gas"),
    ("diesel fuel", "Transportation", "Fuel/Gas"),
    ("vehicle maintenance", "Transportation", "Vehicle Maintenance"),
    ("car service", "Transportation", "Vehicle Maintenance"),
    ("vehicle repair", "Transportation", "Vehicle Maintenance"),
    ("pamasahe", "Transportation", "Public Transit"),
    ("sa bus", "Transportation", "Public Transit"),
    ("sa jeep", "Transportation", "Public Transit"),
    ("sa trike", "Transportation", "Public Transit"),
    ("gasolina", "Transportation", "Fuel/Gas"),
    ("krudo", "Transportation", "Fuel/Gas"),
    ("diesel", "Transportation", "Fuel/Gas"),
    ("change oil", "Transportation", "Vehicle Maintenance"),
    ("talyer", "Transportation", "Vehicle Maintenance"),
    ("mekaniko", "Transportation", "Vehicle Maintenance"),
    ("parking", "Transportation", "Vehicle Maintenance"),

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

    # Entertainment
    ("SM Cinema tickets", "Entertainment", "Movies/Shows"),
    ("Ayala Cinema movie", "Entertainment", "Movies/Shows"),
    ("Steam games purchase", "Entertainment", "Gaming"),
    ("Nintendo Switch game", "Entertainment", "Gaming"),
    ("PlayStation Plus subscription", "Entertainment", "Gaming"),
    ("Xbox Game Pass", "Entertainment", "Gaming"),
    ("MOA concert tickets", "Entertainment", "Concert Tickets/Events"),
    ("Araneta show tickets", "Entertainment", "Concert Tickets/Events"),
    ("BGC art supplies", "Entertainment", "Hobbies"),
    ("National Bookstore art materials", "Entertainment", "Hobbies"),
    ("movie ticket", "Entertainment", "Movies/Shows"),
    ("cinema admission", "Entertainment", "Movies/Shows"),
    ("video game", "Entertainment", "Gaming"),
    ("game subscription", "Entertainment", "Gaming"),
    ("concert ticket", "Entertainment", "Concert Tickets/Events"),
    ("event admission", "Entertainment", "Concert Tickets/Events"),
    ("hobby materials", "Entertainment", "Hobbies"),
    ("craft supplies", "Entertainment", "Hobbies"),
    ("sine", "Entertainment", "Movies/Shows"),
    ("pelikula", "Entertainment", "Movies/Shows"),
    ("concert", "Entertainment", "Concert Tickets/Events"),
    ("pustiso", "Entertainment", "Concert Tickets/Events"),
    ("video games", "Entertainment", "Gaming"),
    ("laro", "Entertainment", "Gaming"),
    ("libangan", "Entertainment", "Hobbies"),
    ("hobby", "Entertainment", "Hobbies")
]

class ExpenseClassifier:
    def __init__(self, training_data):
        """Initialize the classifier with training data"""
        self.training_data = training_data
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3))

        # Prepare training texts and labels
        self.texts = [entry[0] for entry in training_data]
        self.categories = [entry[1] for entry in training_data]
        self.expense_types = [entry[2] for entry in training_data]

        # Fit vectorizer
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)

    def preprocess_text(self, text):
        """Preprocess input text"""
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

    def classify_expense(self, user_input):
        """Classify user input into category and expense type"""
        # Preprocess input
        processed_input = self.preprocess_text(user_input)

        # Vectorize input
        input_vector = self.vectorizer.transform([processed_input])

        # Calculate similarities
        similarities = cosine_similarity(input_vector, self.tfidf_matrix).flatten()

        # Get best match
        best_match_idx = np.argmax(similarities)
        confidence_score = similarities[best_match_idx]

        # Get predicted category and expense type
        predicted_category = self.categories[best_match_idx]
        predicted_expense_type = self.expense_types[best_match_idx]

        return {
            'user_input': user_input,
            'category': predicted_category,
            'expense_type': predicted_expense_type,
            'confidence_score': confidence_score,
            'requires_new_expense_type': confidence_score < 0.3
        }

    def get_similar_entries(self, user_input, n=5):
        """Get n most similar entries for reference"""
        input_vector = self.vectorizer.transform([self.preprocess_text(user_input)])
        similarities = cosine_similarity(input_vector, self.tfidf_matrix).flatten()

        # Get top n matches
        top_indices = similarities.argsort()[-n:][::-1]

        return [
            {
                'text': self.texts[idx],
                'category': self.categories[idx],
                'expense_type': self.expense_types[idx],
                'similarity': similarities[idx]
            }
            for idx in top_indices
        ]

# Initialize classifier
classifier = ExpenseClassifier(training_data)

app = Flask(__name__)
CORS(app)

# Initialize classifier
classifier = ExpenseClassifier(training_data)

@app.route('/classify_expense', methods=['POST'])
def classify_expense():
    data = request.get_json()
    expense_text = data.get('expense_text', '')

    result = classifier.classify_expense(expense_text)
    result['confidence_score'] = float(result['confidence_score'])
    result['requires_new_expense_type'] = bool(result['requires_new_expense_type'])
    
    similar_entries = classifier.get_similar_entries(expense_text)
    for entry in similar_entries:
        entry['similarity'] = float(entry['similarity'])

    return jsonify({
        'classification': result,
        'similar_entries': similar_entries
    })

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    # Download NLTK data
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)




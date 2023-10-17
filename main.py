import os
import nltk
nltk.data.path.append(r"D:\nltk_data-gh-pages\packages")
# nltk.download('cmudict')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import cmudict
from textblob import TextBlob
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


class TextProcessor:

    def __init__(self,input_dir, stop_words_dir, article_dir, cleaned_article_dir, master_dict_dir):
        self.input_urls_dir = input_dir
        self.stop_words_dir = stop_words_dir
        self.article_dir = article_dir
        # self.cleaned_article_dir = os.path.join(os.getcwd(), r'\cleaned_data')
        self.master_dict_dir = master_dict_dir
        self.cmu = cmudict.dict()
        self.cleaned_article_dir = cleaned_article_dir


    def get_stopwords(self):
        stop_words = set()
        stop_words_files = os.listdir(self.stop_words_dir)        
        for stop_words_file in stop_words_files:
            with open(os.path.join(self.stop_words_dir, stop_words_file), 'r', encoding='utf-8') as file:
                stop_words.update(line.split()[0].strip() for line in file.readlines())
                stop_words = {word.lower() for word in stop_words}
        return stop_words
        

    def clean_article(self, article_text, stop_words):
        words = article_text.split() 
        special_characters = ['.', ',', '!', '?', ';', ':', "'", '"', '-', '–', '...', '(', ')', '[', ']', '{', '}', '+', '-', '*', '/', '%', '$', '@']
        filtered_words = [word.lower() for word in words if word.lower() not in stop_words and word.lower() not in special_characters]
        filtered_text = ' '.join(filtered_words)
        return filtered_text

    def get_articles(self):
        links_df = pd.read_excel(self.input_urls_dir)
        print('Data Extractiion Begin')
        for index, row in tqdm(links_df.iterrows(), total=len(links_df)):
            id = row[0]
            link = row[1]

            try:
                response = requests.get(link)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                # heading of Article
                try:
                    heading = soup.select_one("h1.entry-title, h1.tdb-title-text").get_text()
                except Exception as e:
                    print(f"\n \n No <h1> element with class 'entry-title' found., {id}")
                    print(f"An error occurred: {e}")
                # body of Article
                try:
                    div_element = soup.select_one("div.td-post-content.tagdiv-type, tdb-block-inner.td-fix-index")
                except Exception as e:
                    print(f"\n \n No data found at id= {id}")
                    print(f"An error occurred: {e}")
                    continue
                # saving data
                full_article = div_element.get_text()
                full_article_text = heading + full_article
                file_path = f'D:/URL_LINK/{id}.txt'
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(full_article_text)

            except Exception as e:
                print(f"An error occurred while processing {id}: {e}")
        
        return print("Data Extracted and saved")   



    def clean_articles(self):
        stop_words = self.get_stopwords()
        article_files = os.listdir(self.article_dir)
        print("\nData Cleaning Started...")
        # getting all the articles, cleaning them & storing in txt file
        for article_file in article_files:
            input_file_path = os.path.join(self.article_dir, article_file)
            output_file_path = os.path.join(self.cleaned_article_dir, article_file)
            # opening the article file
            with open(input_file_path, 'r', encoding='utf-8') as input_file:
                article_text = input_file.read()            
            # calling the clean_article function
            cleaned_text = self.clean_article(article_text, stop_words)
            # saving the cleaned article file
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(cleaned_text)
        return print("Data Cleaning Completed and Saved.")


    def _calculate_scores(self, file_name, article):
        url_id = file_name
        full_article = article
        # Tokenize the text into sentences and words
        sentences = sent_tokenize(full_article)
        words = word_tokenize(full_article)
        total_word_count = len(words)
        # calculation of positive and negative word count
        positive_words = self._load_word_list(master_dict_dir, "positive")
        negative_words = self._load_word_list(master_dict_dir, "negative")
        positive_word_count = sum(1 for word in words if word in positive_words)
        negative_word_count = sum(1 for word in words if word in negative_words)
        # Calculate the positive and negative scores as a percentage of total words
        positive_score = (positive_word_count)
        negative_score = (negative_word_count)
        polarity_score = (positive_score - negative_score) / (positive_score + negative_score)
        # subjectivity score
        blob = TextBlob(full_article)
        subjectivity_score = blob.sentiment.subjectivity
        #syllables & complex word scores
        total_count, total_complex_count = self._count_complex_words(words, self.cmu)
        average_syllables_per_word = total_count/ total_word_count
        percentage_complex_words = ( total_complex_count / total_word_count)
        # Calculate the count of personal pronouns
        personal_pronouns = ["i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "we", "us", "our", "ours", "ourselves", "they", "them", "their", "theirs", "themselves"]
        personal_pronoun_count = sum(1 for word in words if word.lower() in personal_pronouns)
        # other scores
        average_sentence_length = total_word_count / len(sentences)
        fog_index = 0.4 * (average_sentence_length + percentage_complex_words)
        avg_number_words_sentence = total_word_count / len(sentences)
        average_word_length = sum(len(x) for x in words) / total_word_count

        # creating a dataframe of scores
        df_data = {
                'URL_ID': [url_id],
                'POSITIVE SCORE': [positive_score],
                'NEGATIVE SCORE': [negative_score],
                'POLARITY SCORE': [polarity_score],
                'SUBJECTIVITY SCORE': [subjectivity_score],
                'AVG SENTENCE LENGTH': [average_sentence_length],
                'PERCENTAGE OF COMPLEX WORDS': [percentage_complex_words],
                'FOG INDEX': [fog_index],
                'AVG NUMBER OF WORDS PER SENTENCE': [avg_number_words_sentence],
                'COMPLEX WORD COUNT': [percentage_complex_words],
                'WORD COUNT': [total_word_count],
                'SYLLABLE PER WORD': [average_syllables_per_word],
                'PERSONAL PRONOUNS': [personal_pronoun_count],
                'AVG WORD LENGTH': [average_word_length]
            }

        df = pd.DataFrame(data=df_data)
        return df


    def _load_word_list(self, file_path, count_for):
        if count_for == "positive":
            pos_file_path = r'\positive-words.txt'
            with open(file_path+pos_file_path, 'r', encoding='utf-8') as file:
                word_list = [line.strip().lower() for line in file]           
            return word_list
        
        elif count_for == 'negative':
            neg_file_path = r'\negative-words.txt'
            with open(file_path+neg_file_path, 'r', encoding='utf-8') as file:
                word_list = [line.strip().lower() for line in file]            
            return word_list        


    def _count_complex_words(self, words, cmu):
        total_complex_count = 0
        total_count = 0
        for word in words:         
            if word.lower() in cmu:
                # Get the pronunciation for the word
                pronunciation = cmu[word.lower()][0]  # Use the first pronunciation if multiple exist
                # Count phonemes that end with a digit (indicating stress)
                syllable_count = len([ph for ph in pronunciation if ph[-1].isdigit()])
                total_count += syllable_count
                if syllable_count > 2:
                    total_complex_count +=1
        return total_count, total_complex_count


    def calculate_and_save_score(self):
        df_columns = ['URL_ID', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE',
                    'SUBJECTIVITY SCORE', 'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS',
                    'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT',
                    'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH']
        
        final_df = pd.DataFrame(columns=df_columns)
        # directories url_data
        current_directory = os.getcwd()
        file_location_url_data = os.path.join(current_directory, r'D:\Output_Data_Structure.xlsx')
        url_data_df = pd.read_excel(file_location_url_data, sheet_name='Sheet1')
        url_df = url_data_df[['URL_ID', 'URL']]
        print("\nCalculating Variables:-")
        print("Loading...")
        # Score calculation for each article
        cleaned_articles = os.listdir(self.article_dir)                     
        for cleaned_article in cleaned_articles:
            with open(os.path.join(cleaned_article_dir, cleaned_article), 'r', encoding='utf-8') as file:
                full_article = file.read()
            file_name = cleaned_article.replace(".txt", "")
            # calling function for score calculation, which returns a dataframe
            scores_df = self._calculate_scores(file_name, full_article)
            final_df = pd.concat([final_df, scores_df]).reset_index(drop=True)

        # merging the data to get ulrs & saving it to final_data folder
        print("Calculation Done, Now Saving the DATA...")
        url_df.loc[ :,'URL_ID'] = url_df['URL_ID'].astype(str)
        final_df.loc[ :, 'URL_ID'] = final_df['URL_ID'].astype(str)
        merged_df = pd.merge(url_df, final_df, on='URL_ID', how='left')
        print('saving_data')
        
        with open(os.path.join(current_directory, r'D:/Output_Data_Structure.csv'), 'w', encoding='utf-8', newline='') as file:
                merged_df.to_csv(file, index=False)
        print("all the process is completed and data is saved")

        

       

# Usage example:
input_files_dir = r'D:\Input.xlsx'                        # dir of input files for data extraction(ulrs).
master_dict_dir = r'D:\MasterDictionary'                            # path for matser dir (Negative and Positve wrods)
stop_words_dir = r'D:\StopWords'                                    #dir. where stop words are located
article_dir = r'D:\URL_LINK'                      # dir. where articles are located
cleaned_article_dir = r'D:\cleand_Data'                # dir where cleaned article will be stored

text_processor = TextProcessor(input_files_dir, stop_words_dir, article_dir, cleaned_article_dir, master_dict_dir)

text_processor.get_articles()
text_processor.clean_articles()
text_processor.calculate_and_save_score()




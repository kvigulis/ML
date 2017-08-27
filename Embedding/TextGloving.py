import numpy as np
import pandas as pd
import multiprocessing
from multiprocessing import Pool

# =======
# Step 1: Modify the Glove file by changing values of two dummy words that are not in the genetic text.
# =======
# Feed glove text file into a pandas dataframe
with open('./DataSet/glove.6B.300d.txt') as file:
     array2d = [[str(digit) for digit in line.split()] for line in file]
# Glove word vector embedding as np array.
glove300 = pd.DataFrame(array2d)
#Change dummy word values to zero:
print(glove300.loc[glove300[0] == 'helicopter'])
glove300.loc[3134,1:300] = '1' # Change all to '1'
print(glove300.loc[glove300[0] == 'helicopter'])
print(glove300.loc[glove300[0] == 'panda'])
glove300.loc[15110,1:300] = '0' # Change all to '0'
print(glove300.loc[glove300[0] == 'panda'])
# Save the modified glove version.
glove300.to_csv("glove300Dummified.csv" , index = False )


# # =======
# # Step 2: Split into columns and cut the end of the text.
# # =======
# # Load the text of the competition
# train_text_df = pd.read_csv("/storage/Linux Files/PycharmProjects/KaggleCancer/DataSet/training_text", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])
# test_text_df = pd.read_csv("/storage/Linux Files/PycharmProjects/KaggleCancer/DataSet/test_text", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])
# # Append Test data to Training data
# full = train_text_df.append( test_text_df , ignore_index = True )
#
# # Clean up the text and split into a columns using 'space' characters.
# text = full[['Text']] # Get just the text
# text = pd.DataFrame(text['Text'].str.lower().str.replace('-', ' ').str.replace('\(|\)|\,|\%|\.|\[|\]|\<|\>|\{|\}|\!|\"|\'', '').str.split(' ').tolist())
# # Cut the text up to some number of words:
# # The time of the 'unknown word' replacement step will greatly vary depending on this length.
# text = text.ix[:,0:2000]
# print(text)
# # Save the cleaned text.
# text.to_csv( "/storage/Linux Files/PycharmProjects/KaggleCancer/text_shorter2000.csv" , index = False )


# =======
# Step 3: Replace the Nans and words not in the Glove with 'helicopter' and 'panda'.
# =======
# text = pd.read_csv("/storage/Linux Files/PycharmProjects/KaggleCancer/text_shorter2000.csv")
# glove300 = pd.read_csv("/storage/Linux Files/PycharmProjects/KaggleCancer/glove300Dummified.csv")
# dummy_word1 = 'helicopter' # To replace all the NaNs at the ends of shorter texts.
# dummy_word2 = 'panda' # To replace all the words which are not in the GloVe set.
#
# # Replace all the NaNs at the ends of shorter texts.
# text = text.fillna(dummy_word1)
# # Replace all the words which are not in the GloVe set with dummy_word2.
# # Very long step...
# text1 = pd.DataFrame(text[:450])
# text2 = pd.DataFrame(text[450:900])
# text3 = pd.DataFrame(text[900:1350])
# text4 = pd.DataFrame(text[1350:1800])
# text5 = pd.DataFrame(text[1800:2250])
# text6 = pd.DataFrame(text[2250:2700])
# text7 = pd.DataFrame(text[2700:3150])
# text8 = pd.DataFrame(text[3150:3600])
# text9 = pd.DataFrame(text[3600:4050])
# text10 = pd.DataFrame(text[4050:4500])
# text11 = pd.DataFrame(text[4500:4950])
# text12 = pd.DataFrame(text[4950:5400])
# text13 = pd.DataFrame(text[5400:5850])
# text14 = pd.DataFrame(text[5850:6300])
# text15 = pd.DataFrame(text[6300:6750])
# text16 = pd.DataFrame(text[6750:7200])
# text17 = pd.DataFrame(text[7200:7650])
# text18 = pd.DataFrame(text[7650:8100])
# text19 = pd.DataFrame(text[8100:8550])
# text20 = pd.DataFrame(text[8550:])
#
# texts = [text1, text2, text3, text4, text5, text6, text7, text8, text9, text10, text11, text12, text13, text14, text15, text16, text17, text18, text19, text20]
#
# def replace_with_dummy(text):
#     for index, row in text.iterrows():
#         print(index)
#         for word in row.iteritems(): # Very long looooooooooop...........
#             if word[1] not in glove300['0'].values: # ...and expensive operation.
#                 #print("replacement happened...")
#                 text.loc[index, word[0]] = dummy_word2 # dummy word
#     return text
#
# pool = Pool()
# result = pool.map(replace_with_dummy, texts)
# pool.close()
# pool.join()
#
# text = pd.concat(result)
#
#
# # Save the text - ready for Glovification.
# text.to_csv("/storage/Linux Files/PycharmProjects/KaggleCancer/text_ready_for_glove_mp2000.csv", index=False)


# # =======
# # Step 4: Glovifiy the text... I.e replace the word with Glove vector embeddings and save as 3D np array
# # =======
text = pd.read_csv("/storage/Linux Files/PycharmProjects/KaggleCancer/text_ready_for_glove_mp2000.csv")
glove300 = pd.read_csv("/storage/Linux Files/PycharmProjects/KaggleCancer/glove300Dummified.csv")

text1 = pd.DataFrame(text[:450])
text2 = pd.DataFrame(text[450:900])
text3 = pd.DataFrame(text[900:1350])
text4 = pd.DataFrame(text[1350:1800])
text5 = pd.DataFrame(text[1800:2250])
text6 = pd.DataFrame(text[2250:2700])
text7 = pd.DataFrame(text[2700:3150])
text8 = pd.DataFrame(text[3150:3600])
text9 = pd.DataFrame(text[3600:4050])
text10 = pd.DataFrame(text[4050:4500])
text11 = pd.DataFrame(text[4500:4950])
text12 = pd.DataFrame(text[4950:5400])
text13 = pd.DataFrame(text[5400:5850])
text14 = pd.DataFrame(text[5850:6300])
text15 = pd.DataFrame(text[6300:6750])
text16 = pd.DataFrame(text[6750:7200])
text17 = pd.DataFrame(text[7200:7650])
text18 = pd.DataFrame(text[7650:8100])
text19 = pd.DataFrame(text[8100:8550])
text20 = pd.DataFrame(text[8550:])

texts = [text1, text2, text3, text4, text5, text6, text7, text8, text9, text10, text11, text12, text13, text14, text15, text16, text17, text18, text19, text20]

# Glovifing...
def glovify(text):
    embedded_subarray = np.zeros((text.shape[0],text.shape[1], 300), dtype=float)
    first_index = text.index[0]
    print("first index", first_index)
    print("shape: ", embedded_subarray.shape)
    print("text rows", text.shape[0])
    print("text columns", text.shape[1])

    for index, row in text.iterrows():

        print(index)
        for word in row.iteritems():
            embedding = glove300.loc[glove300['0'] == word[1]].iloc[0,1:]
            embedding = embedding.as_matrix()
            embedded_subarray[int(index)-first_index, int(word[0])] = embedding


    return embedded_subarray

pool = Pool()
result = pool.map(glovify, texts)
pool.close()
pool.join()


embedded_array = np.concatenate(result , axis=0)

print(embedded_array)

np.save('embedded_array2000.npy', embedded_array)
import streamlit as st
import keras_nlp
import json
import pickle
from tensorflow import keras
import tensorflow as tf

BATCH_SIZE = 64
MAX_SEQUENCE_LENGTH = 70



with open('eng_tokenizer.pickle', 'rb') as handle:
    eng_tokenizer = pickle.load(handle)
with open('pdg_tokenizer.pickle', 'rb') as handle:
    pdg_tokenizer = pickle.load(handle)

@st.cache(allow_output_mutation=True)
def cached_model():
    model = keras.models.load_model('my_model_eng_pdg.h5')
    return model


transformer = cached_model()

st.header('ENglish To Piding Machine translation')
st.markdown("Enter an english sentence to see the translation")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []


with st.form('form', clear_on_submit=True):
    user_input = st.text_input('enter enlish sentence: ', '')
    submitted = st.form_submit_button('Translate')




def decode_sequences(input_sentences):
    batch_size = tf.shape(input_sentences)[0]

    # Tokenize the encoder input.
    encoder_input_tokens = eng_tokenizer(input_sentences).to_tensor(
        shape=(None, MAX_SEQUENCE_LENGTH)
    )

    # Define a function that outputs the next token's probability given the
    # input sequence.
    def token_probability_fn(decoder_input_tokens):
        return transformer([encoder_input_tokens, decoder_input_tokens])[:, -1, :]

    # Set the prompt to the "[START]" token.
    prompt = tf.fill((batch_size, 1), pdg_tokenizer.token_to_id("[START]"))

    generated_tokens = keras_nlp.utils.greedy_search(
        token_probability_fn,
        prompt,
        max_length=40,
        end_token_id=pdg_tokenizer.token_to_id("[END]"),
    )
    generated_sentences = pdg_tokenizer.detokenize(generated_tokens)
    #print(generated_sentences)
    return generated_sentences








if submitted and user_input:


    input_sentence = random.choice(user_input.lower())
    translated = decode_sequences(tf.constant([input_sentence]))
    translated = translated.numpy()[0].decode("utf-8")
    translated = (
        translated.replace("[PAD]", "")
        .replace("[START]", "")
        .replace("[END]", "")
        .strip()
    )
    st.session_state.generated.append(translated)

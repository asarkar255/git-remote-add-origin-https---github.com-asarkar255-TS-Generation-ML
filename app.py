import streamlit as st
import os
import json
from datasets import Dataset
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    Trainer, TrainingArguments, DataCollatorForSeq2Seq
)
from sklearn.model_selection import train_test_split
from utils import extract_text_from_docx

st.set_page_config(page_title="ABAP TSD Generator", layout="wide")
st.title("📘 ABAP → Technical Specification Generator")

mode = st.radio("Choose Mode", ["🧠 Train Model", "📄 Generate TSD"])

if mode == "🧠 Train Model":
    st.markdown("### Upload ABAP Code Files (.txt) and Corresponding TSD Files (.docx)")
    uploaded_abap_files = st.file_uploader("Upload ABAP .txt files", accept_multiple_files=True, type="txt")
    uploaded_tsd_files = st.file_uploader("Upload TSD .docx files", accept_multiple_files=True, type="docx")

    if st.button("Train Model"):
        if not uploaded_abap_files or not uploaded_tsd_files:
            st.error("Please upload both ABAP and TSD files.")
        elif len(uploaded_abap_files) != len(uploaded_tsd_files):
            st.error("Mismatch in number of ABAP and TSD files.")
        else:
            data_pairs = []
            for abap_file, tsd_file in zip(uploaded_abap_files, uploaded_tsd_files):
                abap_text = abap_file.read().decode("utf-8")
                tsd_text = extract_text_from_docx(tsd_file)
                data_pairs.append({"input_text": abap_text, "target_text": tsd_text})

            texts = [item["input_text"] for item in data_pairs]
            labels = [item["target_text"] for item in data_pairs]
            train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

            train_dataset = Dataset.from_dict({"input_text": train_texts, "target_text": train_labels})
            val_dataset = Dataset.from_dict({"input_text": val_texts, "target_text": val_labels})

            model_name = "t5-small"
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name)

            def preprocess(example):
                input_text = "translate abap to tsd: " + example["input_text"]
                model_input = tokenizer(input_text, max_length=512, padding="max_length", truncation=True)
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(example["target_text"], max_length=128, padding="max_length", truncation=True)
                model_input["labels"] = labels["input_ids"]
                return model_input

            tokenized_train = train_dataset.map(preprocess)
            tokenized_val = val_dataset.map(preprocess)

            training_args = TrainingArguments(
                output_dir="models",
                evaluation_strategy="epoch",
                learning_rate=2e-4,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                num_train_epochs=3,
                weight_decay=0.01,
                logging_dir="logs",
                save_strategy="epoch",
            )

            data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_val,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )

            with st.spinner("Training the model... This might take a few minutes."):
                trainer.train()
                model.save_pretrained("models")
                tokenizer.save_pretrained("models")
                st.success("Model trained and saved successfully!")

elif mode == "📄 Generate TSD":
    st.markdown("### Upload ABAP Code File (.txt)")
    uploaded_file = st.file_uploader("Upload ABAP .txt file", type="txt")

    if uploaded_file is not None:
        abap_text = uploaded_file.read().decode("utf-8")

        try:
            model = T5ForConditionalGeneration.from_pretrained("models")
            tokenizer = T5Tokenizer.from_pretrained("models")

            input_text = "translate abap to tsd: " + abap_text
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
            outputs = model.generate(**inputs, max_length=200)
            tsd_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

            st.markdown("### Generated Technical Specification")
            st.text_area("TSD Output", tsd_output, height=300)
        except:
            st.error("Model not found. Please train the model first.")

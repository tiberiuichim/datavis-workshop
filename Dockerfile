FROM floydhub/textacy:latest

RUN python -m spacy.en.download all

RUN pip --no-cache-dir install \
  pyLDAvis

ADD eea_corpus.py .
ADD load_eea_corpus.py .
ADD vis.py .
ADD data.csv .

EXPOSE 8888

CMD python load_eea_corpus.py

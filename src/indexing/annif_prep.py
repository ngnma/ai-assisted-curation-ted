import pandas as pd
from src.indexing.data import load_indexing_data, load_vocab_uris
from src.config import ROOT

def create_line(text, labels, tag_to_uri):
    # clean the text and store in a tsv format for Annif: text \t <tag1> <tag2> ...
    cleaned_text = " ".join(text.split())
    mapped_labels = [tag_to_uri[label] for label in labels]
    return f"{cleaned_text}\t{'<' + '> <'.join(mapped_labels) + '>'}"

def save_to_tsv(df, filename, tag_to_uri):
    """Save a DataFrame to a TSV file for Annif: one line per talk, with text and gold labels. (For train, val, train+val sets in training phase)"""
    lines = df.apply(lambda row: create_line(row['text'], row['labels'], tag_to_uri), axis=1).to_list()
    content = "\n".join(lines) + "\n"

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

def save_talks(df, folder):
    """Save a DataFrame to a folder of .txt files for Annif: one file per talk, named by talk_id. (For test, val sets in prediction phase)"""
    folder.mkdir(parents=True, exist_ok=True)   # create the folder if it doesn't exist
    for old_file in folder.glob("*.txt"):       # remove talks from earlier runs
        old_file.unlink()
    
    for i in range(len(df)):
        talk_id = df.loc[i, 'talk_id']
        text = df.loc[i, 'text']
        cleaned_text = " ".join(text.split())
        # save line to txt file
        with open(folder / f'{talk_id}.txt', 'w', encoding='utf-8') as f:
            f.write(cleaned_text)

def check_tsv(filename, df, uri_to_tag):
    """Check a training TSV: one line per talk, exactly 2 columns, every subject is a known tag URI."""
    with open(filename, encoding='utf-8') as f:
        lines = f.read().split("\n")[:-1]  # the file ends with "\n", so the last piece is empty

    assert len(lines) == len(df), f"{filename.name}: {len(lines)} lines, expected {len(df)} talks"
    for n, line in enumerate(lines, start=1):
        parts = line.split("\t")
        assert len(parts) == 2, f"{filename.name} line {n}: {len(parts)} columns, expected 2"
        text, subjects = parts
        assert text, f"{filename.name} line {n}: empty text"
        uris = [s.strip("<>") for s in subjects.split()]
        assert uris, f"{filename.name} line {n}: no subjects"
        unknown = [u for u in uris if u not in uri_to_tag]
        assert not unknown, f"{filename.name} line {n}: unknown URIs {unknown}"
    print(f"{filename.name}: {len(lines)} lines OK")

def check_folder(folder, df):
    """Check a prediction folder: one non-empty .txt per talk, named by talk_id, with no gold labels inside."""
    files = list(folder.glob("*.txt"))
    assert len(files) == len(df), f"{folder.name}/: {len(files)} files, expected {len(df)} talks"
    assert {f.stem for f in files} == set(df['talk_id']), f"{folder.name}/: file names don't match talk_ids"
    for f in files:
        text = f.read_text(encoding='utf-8')
        assert text.strip(), f"{f.name} is empty"
        assert "<https://" not in text, f"{f.name} contains a subject URI (gold labels leaked into the text)"
    print(f"{folder.name}/: {len(files)} files OK")


if __name__ == "__main__":
    # load the vocabulary mapping: tag name <-> URI mapping
    tag_to_uri, uri_to_tag, group_uris = load_vocab_uris()

    # load the data splits
    splits = load_indexing_data()
    train, test, val = splits['train'], splits['test'], splits['val']
    train_val = pd.concat([train, val])

    # save the data in TSV format for Annif
    save_to_tsv(train, ROOT / 'annif/corpus/train.tsv', tag_to_uri)
    save_to_tsv(val, ROOT / 'annif/corpus/val.tsv', tag_to_uri)
    save_to_tsv(train_val, ROOT / 'annif/corpus/train_val.tsv', tag_to_uri)

    # prediction
    save_talks(test, ROOT / 'annif/corpus/test/')
    save_talks(val, ROOT / 'annif/corpus/val/')

    # checks: stop loudly if any file is wrong
    check_tsv(ROOT / 'annif/corpus/train.tsv', train, uri_to_tag)
    check_tsv(ROOT / 'annif/corpus/val.tsv', val, uri_to_tag)
    check_tsv(ROOT / 'annif/corpus/train_val.tsv', train_val, uri_to_tag)
    check_folder(ROOT / 'annif/corpus/test/', test)
    check_folder(ROOT / 'annif/corpus/val/', val)
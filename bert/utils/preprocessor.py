import re
from pathlib import Path


def clean_text(fulltxt_str):

    """Clean the full text of web page into the string for tokenization.

    Args:
        fulltxt_str: A string for cleaning.
    Returns:
        A processed string.
    """

    # Patterns of front matters from differnt universities.
    front_matter = r'^(.|\n)*(GMT|Content-length: [0-9]*|Content-type: text/html)'
    # Patterns of HTML elements between "<>" as well as special tokens.
    html_elem = r'<(.|\n)*?>|[\t\n\.\?]'

    res = re.sub(front_matter, ' ', fulltxt_str)
    res = re.sub(html_elem, ' ', res)
    res = re.sub('  *', ' ', res)
    return res



def preprocess(dataset_dir='./webkb', targ_dir='./'):

    """Pre-process original WebKb dataset.
    The pre-processed dataset will be genrated and stored to the given path 
    as a TSV file named 'dataset.tsv', with each line of record
    in 'uni_name-category_name-text-url' split by tab.
    Please make sure the directory of the dataset is given or './webkb' is in 
    current working directory. 

    Args:
        dataset_dir: A string of path points to the original dataset directory.
            Default as './webkb'.
        targ_dir: A string of path points to the target directory for 
            storing pre-processed dataset. Default as './'
    """

    dataset_dir = Path(dataset_dir)
    cat_lt = cat_lt=[
        'student', 'faculty', 'course', 'project', 
        'department', 'staff', 'other']
    uni_lt = ['cornell', 'texas', 'wisconsin', 'washington', 'misc']
    with open(targ_dir + '/dataset.tsv', 'w+',encoding='utf-8') as targ_file:
        for cat_name in cat_lt:
            print(f'Pre-processing [{cat_name:10}] from', end='')
            for uni_name in uni_lt:
                print(f' [{uni_name}]', end='')
                for text_file in (dataset_dir/cat_name/uni_name).iterdir():
                    text = clean_text(text_file.read_text(errors='replace'))
                    url = (text_file.name.replace('http_', 'http:')
                        .replace('^', '/').lower().rstrip('/'))
                    new_line = '\t'.join(
                        [uni_name, cat_name, text, url]) + '\n'
                    targ_file.write(new_line)
            print('')
    print('Pre-processing is finished.')



if __name__ == '__main__':
    preprocess('./webkb', '.')
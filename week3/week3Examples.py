import os

recall_totals = {
    'TVs': 89365,
    'Laptop & Netbook Computers': 121497,
    'Music': 103060
}

regex_filters = {
    'TVs': 'television\|tv',
    'Laptop & Netbook Computers': 'laptop\|netbook',
    'Music': 'music\|rock\|hip hop'
}


for k in regex_filters.keys():
    cmd = f"grep -i '{regex_filters[k]}' /workspace/datasets/train.csv | cut -d',' -f3 | python leavesToPaths.py --max_depth 3 | sort | uniq -c | sort -nr | head"
    counts = {}
    with os.popen(cmd) as pse:
        for line in pse:
            count = line.split()[0].strip()
            category = line.split('>')[-1].strip()
            counts[category] = int(count)

    precision = counts[k] / sum(counts.values())
    recall = counts[k] / recall_totals[k]
    print(f"{k}: precision: {precision}, recall: {recall}")

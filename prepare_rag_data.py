{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "b66cabbb-bc8d-4b28-85c8-2649c8eaa5ce",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ RAG dataset created from classification dataset\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "# Load dataset\n",
    "df = pd.read_csv(\"data.csv\", encoding=\"ISO-8859-1\", names=[\n",
    "    \"target\", \"ids\", \"date\", \"flag\", \"user\", \"text\"\n",
    "])\n",
    "\n",
    "# Keep only text + label\n",
    "df = df[[\"text\", \"target\"]]\n",
    "\n",
    "# Convert labels\n",
    "df[\"target\"] = df[\"target\"].replace({0: \"Negative\", 4: \"Positive\"})\n",
    "\n",
    "# Take a balanced subset (VERY IMPORTANT)\n",
    "pos = df[df[\"target\"] == \"Positive\"].sample(1000, random_state=42)\n",
    "neg = df[df[\"target\"] == \"Negative\"].sample(1000, random_state=42)\n",
    "\n",
    "rag_df = pd.concat([pos, neg])\n",
    "\n",
    "# Convert to natural language knowledge\n",
    "with open(\"rag_dataset.txt\", \"w\", encoding=\"utf-8\") as f:\n",
    "    for _, row in rag_df.iterrows():\n",
    "        f.write(\n",
    "            f\"Example Tweet: {row['text']}\\n\"\n",
    "            f\"Sentiment: {row['target']}\\n\\n\"\n",
    "        )\n",
    "\n",
    "print(\"✅ RAG dataset created from classification dataset\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8a2b5f28-eeca-4cc2-8290-bc56cf810c66",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

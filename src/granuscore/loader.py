"""Module for reading files."""
import csv
import json
import os
from pathlib import Path


class Reader:
    """General file reader."""

    def __init__(self, encoding="utf-8"):
        self.enc = encoding

    def read(self, file):
        """Read a file."""

        # if already an open file object:
        if hasattr(file, "read"):
            return self.process(file)

        # otherwise treat as file path:
        path = Path(file)
        if not path.is_file():
            return None

        with open(file, "r", encoding=self.enc) as f:
            return self.process(f)

    def write(self, file, lines, mode='a'):
        """Write lines to file."""
        if os.path.dirname(file):
            os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, mode, encoding=self.enc) as f:
            self._write(f, lines)

    def _write(self, file, lines):
        """Write lines to an opened file."""

    def process(self, file):
        """Process an opened file."""


class JSONLineReader(Reader):
    """Reader for .jsonl files."""

    def __init__(self, pretty_print: bool = False):
        super().__init__()
        self.pretty_print = pretty_print

    def process(self, file):
        """Read each line as json object."""
        data = []
        for line in file.readlines():
            try:
                data.append(json.loads(line.strip()))
            except json.decoder.JSONDecodeError:
                pass

        return data

    def _write(self, file, lines):
        for line in lines:
            if self.pretty_print:
                json.dump(line, file, indent=2, ensure_ascii=False)
            else:
                json.dump(line, file, ensure_ascii=False)
            file.write('\n')


class JSONReader(Reader):
    """Reader for .json files."""

    def __init__(self, pretty_print: bool = True):
        super().__init__()
        self.pretty_print = pretty_print

    def process(self, file):
        """Read file as json object."""
        return json.load(file)

    def _write(self, file, dictionary):
        if self.pretty_print:
            json.dump(dictionary, file, indent=2)
        else:
            json.dump(dictionary, file)


class LineReader(Reader):
    """Line reader for files."""

    def process(self, file):
        """Read each line as an entry in a list."""
        data = []
        for line in file.readlines():
            data.append(line)

        return data

    def _write(self, file, lines):
        for line in lines:
            file.write(line)
            file.write('\n')

class TSVReader(Reader):
    """Reader for .tsv files."""

    def __init__(self, delimiter='\t'):
        super().__init__()
        self.delimiter = delimiter

    def process(self, file):
        """Read each row of the TSV as a list of values."""
        reader = csv.reader(file, delimiter=self.delimiter)
        return [row for row in reader]

    def _write(self, file, rows):
        writer = csv.writer(file, delimiter=self.delimiter)
        writer.writerows(rows)
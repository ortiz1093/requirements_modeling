# Requirements Modeling

## Download and Installation
*Only the SpecRelations folder is needed for usage. All other files are deprecated and their functionality is being
migrated to the SpecRelations files. Once migration is complete, they will be removed from the repo.

Pip is not currently supported. Therefore, to install the package locally, either download the repo as zip file or clone
to local machine (green button at top of GitHub repo page). DO NOT PUT ENTIRE REPO IN PYTHON PATH.

Once downloaded/cloned, add /path/to/SpecRelations to your PYTHON PATH. It is recommended that a virtual environment be
used to prevent dependency conflicts.

## Usage
The ``SpecRelations`` package is based around the ``System`` class. This stores all attributes associated with the
system being designed. To create a ``System`` instance, a name must be provided, along with a requirements document.

At the moment, the software treats a requirements document as immutable. There are not currently methods for modifying
the document within the same session. The assumption is made that the package is used to analyze graphs for a given
document, not to modify the document.

In initializing the ``System`` instance, the provided document will be parsed and processed automatically so that the
graphs and other attibutes may all be interacted with immediately. Multiprocessing is leveraged to reduce the run time
of this instantiation. Progress is output to the terminal during this operation.

### Creating a ``System`` instance
The program:
```Python
from SpecRelations import system
filepath = ("FMTV_Requirements_partial.txt")
with open(filepath, "r") as f:
    doc_txt = f.read()
vehicle = system.System("Vehicle", doc_txt)
```

Outputs (partial):
```
Processing section 3.1. System Definition.
Processing section 3.2. Vehicle Characteristics.
Processing section 3.2.1. Performance Characteristics.
Processing section 3.2.1.1. Grade Operation.
...
```

Now we can view the list of requirements using:
```Python
vehicle.print_requirements_list()
```

Or we can view the document structure using:
```Python
vehicle.print_document_tree()
```

Both methods print the output to the terminal.


### Viewing Graphs
Once the system is instantiated, visualize the graphs with:
```Python
vehicle.show_graphs()
```

This defaults to displaying all available relationship graphs and they will all be fully connected.

To display the graph for a subset of the available relationships, use the keyword argument ``relations``:
```Python
vehicle.show_graphs(relations=['keyword', 'similarity'])
```

For a full list of available relations, use ```print_relation_types```:

```Python
vehicle.print_relation_types()
```



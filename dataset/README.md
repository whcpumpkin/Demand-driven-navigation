# Here is the detailed description and the structure of the files in the dataset.

## Description
- instruction.json: the full demand instructions and the names of the objects to which they correspond.
- instruction_small.json: the demand instructions used for training in our paper.
- instruction_unseen.json: the demand instructions used for testing in our paper.

- answer.json: the names of all the objects and the demand instructions they can satisfy.
- answer_small.json: the names of the objects used for training in our paper.
- answer_unseen.json: the names of the objects used for test in our paper.

- env/house_idx_obj_{train/val/test}.json: Describes objects that exist in a particular house.
- env/obj_houst_idx_{train/val/test}.json: Describes the houses in which an object will appear.
- env/{train/val/test}.jsonl.gz: ProcThor dataset

- bc_train_{0,1,2,3,4}_pre.h5: pre-generaged dataset, including CLIP-Visual-features. 
- bc_train_check.json: trajectory metadatam, including paths, instructions, action list, etc. For action list, the mapping is {0: 'MoveAhead', 1: 'RotateLeft', 2: 'RotateRight', 3: 'LookUp', 4: 'LookDown', 5: 'Done'}


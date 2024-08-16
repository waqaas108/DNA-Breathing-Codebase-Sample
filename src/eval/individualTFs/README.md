So the structure of seq_feat is:

partition:seq_id:feature:value

where label is a feature and has as many values as there are cell lines.

To make a TF specific dataset, I need to match a sequence with the label information of all cell lines in that TF group.

Pick all sequences with a positive label for the positive set, and use sequences with no positive labels for the negative set.

Use a set random seed and then pick out random sequences from the negatives.

Build the datasets using the one .pkl file on the spot when beginning to train all the models, somehow make a single file that is a composite of all the checkpoints of the trained models.

For all that, we need to make a modified trainer python file that takes the .pkl file, recodes it, and then runs the model in a loop until it has the 64 trained models.

---

I am starting with the datasets, and I'm thinking it will have two components:

a merged pan-dataset, with structure:
seq_id:feature:value

a dictionary to point at sequences for each subdataset, with structure:
TF:partition:seq_id

--

For this, I'll have to make the merger and then the dictionary generator.

---

I'm thinking about how this model looks like when I run it:

>I open the trainer in terminal, and specify the data_dir, out of which it picks up seq_breathing_feat.pkl, since we specify that in the trainer.

>>Program then picks this file up, runs the TF_sorter on it. I guess it imports the python file for TF_sorter first. TF_sorter now gives us a dataset for every TF that can be picked up.

>>(?) for TF in list(sorted_dataset):
            create a data object named TF
            feed it to the model
            run the rest of the model
            
--

Conclusion I'm coming to is that I should make another python file that calls up the trainer and feeds it the datasets one by one.

So I edit the trainer assuming the datasets it's getting have been modded before.

In which case not much needs to be edited, let's see.
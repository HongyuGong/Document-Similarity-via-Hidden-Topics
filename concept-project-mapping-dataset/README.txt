1. Introduction
This dataset collects science projects from sciencebuddies website and concepts from Next Generation Science Standards (NGSS). A total number of 537 (project, concept) pairs are included, and the task is to decide whether the given project is relevant to a specific concept. If a project matches the given concept, then this pair is a good matching. Otherwise, it's a bad matching. For example, a project studying the relationship between the gravity and the ball's movement is related to the concept of forces in physics, whereas it has nothing to do with the concept of chemical reaction.

2. file structure
	(1) projects.txt: one line for one project, a project has fields including NAME,TITLE, SUBJECT, URL, and CONTENT. Each field is separated by "\t||\t".
	(2) concepts.txt: one line for one concept.
	(3) annotations.txt: a project and a concept in the same line is a project-concept pair, and each line of annotation file gives annotations to each pair. The annotations are binary, "1" corresponds to a good matching, and "0" corresponds to a bad matching.
Each of the three files have 537 lines.

3. Annotations
Undergraduates and graduates from engineering subjects participate in dataset annotations. Each concept-project pair receives at least three annotations, and we take the majority as the matching decision. 
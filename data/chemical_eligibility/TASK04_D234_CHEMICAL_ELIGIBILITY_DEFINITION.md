# D234 molecular-structure eligibility definition

This decision creates a reporting projection over the accepted D115 screening output. It is not a model rerun, does not alter any E005 probability, and does not supersede the D116 screening lock.

A screening record is excluded from the molecular-structure analysis set when its frozen `standardized_smiles` satisfies all of the following under the locked RDKit parser:

1. the structure parses and contains at least one real atom;
2. the molecular graph has zero covalent bonds; and
3. all real atoms have the same atomic number.

This definition captures single-atom elemental records, including charged forms, and any future same-element zero-bond representation. It does not exclude covalently bonded elemental molecules or multi-element salts/mixtures merely because they contain an element or metal.

The complete 5,003-row scan produced 26 matches, all of which contain exactly one atom. There were no parse failures and no zero-bond multi-element records. The 26 records represent 25 distinct element symbols because iron occurs in two frozen forms, `[Fe]` and `[Fe+3]`.

The exclusion is applied only after the immutable predictions have been read. Historical inputs, probabilities, result packages, locks, hashes and run IDs remain unchanged.

**Data Model**
**Drug** is used to refer to a compound or small molecule that can target or interact with an expressed gene protein product.

**Gene** is used to refer to an expressed protein product associated with an existing HGNC gene. Nearly all DGIdb gene records are associated with an HGNC symbol and concept identifiers.

An **interaction** is an observed or inferred interaction between gene products and drug compounds or small molecules. Interactions in DGIdb are imported from published literature or from publicly available databases.

A **category** is an annotation that designates genes that are considered potentially druggable due to emergent properties or structures that are associated with druggability (e.g. kinases, nuclear receptors).


A **claim** is an individual record imported from a source for a single gene, drug, or interaction. This claim contains the record name, aliases, and attributes as reported by the source.

A **group** is a collection of claims that have been linked and ontologically grounded to their underlying concept. Groups are created for gene, drug, and interaction claims. For example, drug claims for Gleevac, Imatininb, and STI-571 that have been imported from different sources would be a group under the central identifier Imatinib.
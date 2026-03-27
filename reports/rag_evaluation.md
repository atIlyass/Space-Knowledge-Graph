# RAG Evaluation, Baseline vs SPARQL-grounded RAG

| # | Question | Baseline Answer | RAG Answer | Retries |
|---|----------|----------------|------------|---------|
| 1 | Who were the astronauts of the Apollo 11 mission? | Neil Armstrong, Buzz Aldrin, Michael Collins |   astronaut: http://www.wikidata.org/entity/Q104859   astronaut: http://space-kg.org/resource/Buzz_Aldrin   astronaut: h | 0 |
| 2 | What is the Hubble Space Telescope part of? | The Hubble Space Telescope is part of NASA's Goddard Space Flight Center program. |   group: http://www.wikidata.org/entity/Q901553   group: http://www.wikidata.org/entity/Q901553 | 0 |
| 3 | Which agency launched the Hubble Space Telescope? | NASA |   agency: http://www.wikidata.org/entity/Q52152   agency: http://www.wikidata.org/entity/Q52152   agency: http://www.wik | 0 |
| 4 | What is the Hubble Space Telescope an instance of? | A space telescope |   class: http://www.wikidata.org/entity/Q148578   class: http://www.wikidata.org/entity/Q148578 | 0 |
| 5 | What countries is SpaceX associated with? | SpaceX is primarily associated with the United States. | (Could not retrieve an answer after 3 retries. Last error: The query executed successfully but returned 0 results.) | 3 |

## Detailed Results

### Q1: Who were the astronauts of the Apollo 11 mission?

**Baseline (no RAG):**
> Neil Armstrong, Buzz Aldrin, Michael Collins

**RAG answer (SPARQL-grounded):**
>   astronaut: http://www.wikidata.org/entity/Q104859
  astronaut: http://space-kg.org/resource/Buzz_Aldrin
  astronaut: http://space-kg.org/resource/Neil_Armstrong

**Generated SPARQL:**
```sparql
PREFIX skg-o: <http://space-kg.org/ontology/>
PREFIX skg-r: <http://space-kg.org/resource/>
SELECT ?astronaut WHERE {
  ?mission skg-o:hasAstronaut ?astronaut .
  FILTER(CONTAINS(LCASE(STR(?mission)), "apollo"))
}
```
*(Repair retries: 0)*

### Q2: What is the Hubble Space Telescope part of?

**Baseline (no RAG):**
> The Hubble Space Telescope is part of NASA's Goddard Space Flight Center program.

**RAG answer (SPARQL-grounded):**
>   group: http://www.wikidata.org/entity/Q901553
  group: http://www.wikidata.org/entity/Q901553

**Generated SPARQL:**
```sparql
PREFIX skg-o: <http://space-kg.org/ontology/>
PREFIX skg-r: <http://space-kg.org/resource/>
SELECT ?group WHERE {
  ?telescope skg-o:partOf ?group .
  FILTER(CONTAINS(LCASE(STR(?telescope)), "hubble"))
}
```
*(Repair retries: 0)*

### Q3: Which agency launched the Hubble Space Telescope?

**Baseline (no RAG):**
> NASA

**RAG answer (SPARQL-grounded):**
>   agency: http://www.wikidata.org/entity/Q52152
  agency: http://www.wikidata.org/entity/Q52152
  agency: http://www.wikidata.org/entity/Q1540619
  agency: http://www.wikidata.org/entity/Q1540619

**Generated SPARQL:**
```sparql
PREFIX skg-o: <http://space-kg.org/ontology/>
PREFIX skg-r: <http://space-kg.org/resource/>
SELECT ?agency WHERE {
  ?telescope skg-o:launchedBy ?agency .
  FILTER(CONTAINS(LCASE(STR(?telescope)), "hubble"))
}
```
*(Repair retries: 0)*

### Q4: What is the Hubble Space Telescope an instance of?

**Baseline (no RAG):**
> A space telescope

**RAG answer (SPARQL-grounded):**
>   class: http://www.wikidata.org/entity/Q148578
  class: http://www.wikidata.org/entity/Q148578

**Generated SPARQL:**
```sparql
PREFIX skg-o: <http://space-kg.org/ontology/>
PREFIX skg-r: <http://space-kg.org/resource/>
SELECT ?class WHERE {
  ?telescope skg-o:instanceOf ?class .
  FILTER(CONTAINS(LCASE(STR(?telescope)), "hubble"))
}
```
*(Repair retries: 0)*

### Q5: What countries is SpaceX associated with?

**Baseline (no RAG):**
> SpaceX is primarily associated with the United States.

**RAG answer (SPARQL-grounded):**
> (Could not retrieve an answer after 3 retries. Last error: The query executed successfully but returned 0 results.)

**Generated SPARQL:**
```sparql
PREFIX skg-o: <http://space-kg.org/ontology/>
PREFIX skg-r: <http://space-kg.org/resource/>
SELECT ?country WHERE {
  ?company skg-o:isAssociatedWith ?country .
  FILTER(CONTAINS(LCASE(STR(?company)), "spacex")) 
  FILTER (CONTAINS(LCASE(STR(?country)), "usa"))
}
```
*(Repair retries: 3)*

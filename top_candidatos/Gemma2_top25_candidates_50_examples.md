# Análise de candidatos — Gemma2

Modelo treinado:
`outputs/Gemma-2-2B-it-4ep-pictogram-id-ft`

Parâmetros da análise:

- Exemplos analisados: 50
- Candidatos por exemplo: top-25
- Conjunto utilizado: `valid.json`
- Tipo de resultado: próximo pictograma previsto

---

## Exemplo 1

**Frase:**   
**Prompt:** `12272 9920 8476 32782 7074`  
**Resposta correta:** ID `12333`  
**Resultado:** a resposta correta apareceu na posição **2**.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 8476 | the | 0.44921875 |  |
| 2 | 12333 | organization | 0.32812500 | ✅ |
| 3 | 36935 | half body shot | 0.10644531 |  |
| 4 | 2627 | one | 0.05346680 |  |
| 5 | 6642 | Poetry and Art's books | 0.01397705 |  |
| 6 | 2704 | city | 0.01196289 |  |
| 7 | 37236 | sincerity | 0.00903320 |  |
| 8 | 6906 | that one | 0.00497437 |  |
| 9 | 11399 | and | 0.00195312 |  |
| 10 | 3322 | jar | 0.00183105 |  |
| 11 | 7091 | that | 0.00167084 |  |
| 12 | 25289 | palate | 0.00157166 |  |
| 13 | 36367 | french class | 0.00151825 |  |
| 14 | 3145 | port | 0.00111389 |  |
| 15 | 38759 | fashion designer | 0.00111389 |  |
| 16 | 37740 | german class | 0.00074005 |  |
| 17 | 31330 | Ceuta | 0.00067520 |  |
| 18 | 36897 | Burgos' Cathedral | 0.00067520 |  |
| 19 | 7194 | for | 0.00059509 |  |
| 20 | 34647 | empathize | 0.00054169 |  |
| 21 | 37544 | parliamentary group | 0.00045013 |  |
| 22 | 16343 | chalks | 0.00032806 |  |
| 23 | 24146 | lancet | 0.00029945 |  |
| 24 | 37250 | videotape | 0.00029945 |  |
| 25 | 14771 | Burgos | 0.00026321 |  |

---

## Exemplo 2

**Frase:**   
**Prompt:** `12272 4725 7034 8476 35431`  
**Resposta correta:** ID `11399`  
**Resultado:** a resposta correta não apareceu no top-25.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 7074 | of | 0.98828125 |  |
| 2 | 36480 | be | 0.00488281 |  |
| 3 | 6493 | disappear | 0.00216675 |  |
| 4 | 7810 | according to | 0.00065994 |  |
| 5 | 5593 | hang out | 0.00056458 |  |
| 6 | 4642 | disappear | 0.00036430 |  |
| 7 | 17322 | always | 0.00035286 |  |
| 8 | 6152 | indigenous person of India | 0.00031090 |  |
| 9 | 35949 | be able to | 0.00015640 |  |
| 10 | 11709 | at the | 0.00011110 |  |
| 11 | 3423 | equal sign | 0.00008154 |  |
| 12 | 32276 | tie up | 0.00005245 |  |
| 13 | 5958 | indigenous person of Asia | 0.00005007 |  |
| 14 | 2746 | listen to music | 0.00004482 |  |
| 15 | 6282 | indigenous person of Africa | 0.00003552 |  |
| 16 | 11712 | of the | 0.00003386 |  |
| 17 | 8474 | a | 0.00003338 |  |
| 18 | 29468 | twelve o'clock | 0.00003076 |  |
| 19 | 30630 | jealous | 0.00003040 |  |
| 20 | 38570 | what is your name? | 0.00003040 |  |
| 21 | 34781 | make fun of | 0.00003040 |  |
| 22 | 7041 | to | 0.00002992 |  |
| 23 | 5367 | next to | 0.00002730 |  |
| 24 | 35757 | dial 061 | 0.00002480 |  |
| 25 | 27408 | caress | 0.00002372 |  |

---

## Exemplo 3

**Frase:**   
**Prompt:** `2630 38296 33970 7041 2704 11399 7041`  
**Resposta correta:** ID `21854`  
**Resultado:** a resposta correta apareceu na posição **2**.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 2704 | city | 0.94921875 |  |
| 2 | 21854 | Statue of Liberty | 0.03051758 | ✅ |
| 3 | 10271 | Eiffel Tower | 0.00823975 |  |
| 4 | 23957 | Rome Coliseum | 0.00341797 |  |
| 5 | 39105 | sweet granadilla | 0.00321960 |  |
| 6 | 21445 | Big Ben | 0.00321960 |  |
| 7 | 19547 | Holland | 0.00118256 |  |
| 8 | 23769 | Atomium | 0.00031853 |  |
| 9 | 32452 | Saint John's day | 0.00024796 |  |
| 10 | 14774 | Salamanca | 0.00012064 |  |
| 11 | 29568 | Holland | 0.00011349 |  |
| 12 | 31386 | Sydney Opera House | 0.00009727 |  |
| 13 | 8058 | Canary Islands | 0.00006247 |  |
| 14 | 27802 | shower cap | 0.00003791 |  |
| 15 | 8112 | United States of America | 0.00003576 |  |
| 16 | 9069 | reindeer | 0.00003242 |  |
| 17 | 14834 | Lerida | 0.00003242 |  |
| 18 | 26667 | Angola | 0.00002456 |  |
| 19 | 5922 | walking stick | 0.00001848 |  |
| 20 | 29468 | twelve o'clock | 0.00001848 |  |
| 21 | 27314 | Vatican City | 0.00001800 |  |
| 22 | 14690 | Seville | 0.00001633 |  |
| 23 | 37544 | parliamentary group | 0.00000888 |  |
| 24 | 23751 | Noah's Ark | 0.00000715 |  |
| 25 | 7238 | Saint Joseph | 0.00000602 |  |

---

## Exemplo 4

**Frase:**   
**Prompt:** `12276 29306 6480 25087 11351 2628 4571`  
**Resposta correta:** ID `27234`  
**Resultado:** a resposta correta não apareceu no top-25.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 36656 | military man | 0.09960938 |  |
| 2 | 35078 | sculptor | 0.07910156 |  |
| 3 | 6642 | Poetry and Art's books | 0.06787109 |  |
| 4 | 14670 | decision | 0.05908203 |  |
| 5 | 24222 | trousers | 0.05468750 |  |
| 6 | 26901 | footsteps | 0.02172852 |  |
| 7 | 6039 | photographic album | 0.02172852 |  |
| 8 | 2565 | trousers | 0.02136230 |  |
| 9 | 32584 | adjectives | 0.02111816 |  |
| 10 | 8076 | scientist | 0.02075195 |  |
| 11 | 37236 | sincerity | 0.01831055 |  |
| 12 | 11398 | somersault | 0.01831055 |  |
| 13 | 38902 | stinger | 0.01721191 |  |
| 14 | 4551 | accident | 0.01696777 |  |
| 15 | 11217 | scientist | 0.01544189 |  |
| 16 | 36297 | monument | 0.01300049 |  |
| 17 | 4631 | school principal | 0.01257324 |  |
| 18 | 2746 | listen to music | 0.00952148 |  |
| 19 | 35081 | sculptor | 0.00866699 |  |
| 20 | 37324 | auditor | 0.00811768 |  |
| 21 | 10304 | countryside | 0.00741577 |  |
| 22 | 14976 | clitoris | 0.00695801 |  |
| 23 | 27327 | surname | 0.00695801 |  |
| 24 | 36249 | historic district | 0.00695801 |  |
| 25 | 2683 | countryside | 0.00674438 |  |

---

## Exemplo 5

**Frase:**   
**Prompt:** `2629 9839 35859 32394 36480 37802 7194`  
**Resposta correta:** ID `3086`  
**Resultado:** a resposta correta não apareceu no top-25.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 12272 | his | 0.31835938 |  |
| 2 | 8476 | the | 0.27343750 |  |
| 3 | 21854 | Statue of Liberty | 0.09277344 |  |
| 4 | 12276 | his | 0.03198242 |  |
| 5 | 12274 | theirs | 0.01611328 |  |
| 6 | 15497 | put the bra on | 0.01464844 |  |
| 7 | 36897 | Burgos' Cathedral | 0.01379395 |  |
| 8 | 8477 | the | 0.00946045 |  |
| 9 | 11712 | of the | 0.00759888 |  |
| 10 | 34781 | make fun of | 0.00738525 |  |
| 11 | 38570 | what is your name? | 0.00714111 |  |
| 12 | 6282 | indigenous person of Africa | 0.00714111 |  |
| 13 | 7264 | put the top on | 0.00631714 |  |
| 14 | 6531 | have an idea | 0.00573730 |  |
| 15 | 34367 | get up from the floor | 0.00540161 |  |
| 16 | 37701 | invade | 0.00540161 |  |
| 17 | 12278 | theirs | 0.00476074 |  |
| 18 | 38202 | clear all button | 0.00476074 |  |
| 19 | 21375 | lean on | 0.00476074 |  |
| 20 | 8555 | turn on the monitor | 0.00448608 |  |
| 21 | 35949 | be able to | 0.00448608 |  |
| 22 | 23751 | Noah's Ark | 0.00448608 |  |
| 23 | 11617 | can you see it? | 0.00360107 |  |
| 24 | 34790 | lend a book | 0.00360107 |  |
| 25 | 6642 | Poetry and Art's books | 0.00347900 |  |

---

## Exemplo 6

**Frase:**   
**Prompt:** `8477 5433 36439 11317`  
**Resposta correta:** ID `29062`  
**Resultado:** a resposta correta não apareceu no top-25.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 11709 | at the | 0.85546875 |  |
| 2 | 36439 | reduce | 0.06396484 |  |
| 3 | 5397 | well | 0.01293945 |  |
| 4 | 38413 | eat with the hand | 0.01110840 |  |
| 5 | 8476 | the | 0.00762939 |  |
| 6 | 7212 | by | 0.00668335 |  |
| 7 | 12274 | theirs | 0.00427246 |  |
| 8 | 7074 | of | 0.00421143 |  |
| 9 | 11712 | of the | 0.00415039 |  |
| 10 | 15497 | put the bra on | 0.00225830 |  |
| 11 | 10177 | sit on the toilet | 0.00218201 |  |
| 12 | 8477 | the | 0.00198364 |  |
| 13 | 36697 | criticise | 0.00181580 |  |
| 14 | 38753 | toggle | 0.00167847 |  |
| 15 | 2675 | drop the glass | 0.00156403 |  |
| 16 | 39069 | prick | 0.00105286 |  |
| 17 | 4714 | attack with a stick | 0.00094223 |  |
| 18 | 6617 | go upstairs | 0.00088501 |  |
| 19 | 12315 | comply | 0.00086975 |  |
| 20 | 2637 | open the box | 0.00078201 |  |
| 21 | 5451 | above | 0.00074387 |  |
| 22 | 8555 | turn on the monitor | 0.00069809 |  |
| 23 | 6037 | drown | 0.00058746 |  |
| 24 | 37485 | accompany | 0.00057983 |  |
| 25 | 37217 | mix with | 0.00054550 |  |

---

## Exemplo 7

**Frase:**   
**Prompt:** `7253 8477 29306 37752 25752 7194 8476`  
**Resposta correta:** ID `11205`  
**Resultado:** a resposta correta não apareceu no top-25.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 37544 | parliamentary group | 0.76171875 |  |
| 2 | 38202 | clear all button | 0.15917969 |  |
| 3 | 38683 | presenter | 0.05029297 |  |
| 4 | 3208 | swimmer | 0.00405884 |  |
| 5 | 7074 | of | 0.00194550 |  |
| 6 | 36387 | elf | 0.00171661 |  |
| 7 | 4604 | clear | 0.00154114 |  |
| 8 | 28354 | swear | 0.00151825 |  |
| 9 | 38201 | clear all button | 0.00128937 |  |
| 10 | 7116 | people | 0.00085831 |  |
| 11 | 36345 | community association | 0.00077438 |  |
| 12 | 8256 | newly married couple | 0.00074005 |  |
| 13 | 12333 | organization | 0.00061417 |  |
| 14 | 5377 | America | 0.00054932 |  |
| 15 | 6039 | photographic album | 0.00048065 |  |
| 16 | 9072 | pick up a package | 0.00046158 |  |
| 17 | 24783 | audience | 0.00044632 |  |
| 18 | 27329 | weekend | 0.00043869 |  |
| 19 | 6564 | see | 0.00042725 |  |
| 20 | 10369 | what program do you want to watch? | 0.00042343 |  |
| 21 | 37876 | producer | 0.00034523 |  |
| 22 | 11698 | produce | 0.00033951 |  |
| 23 | 8207 | meeting | 0.00028229 |  |
| 24 | 29843 | model agency | 0.00026703 |  |
| 25 | 26380 | reproduction | 0.00024605 |  |

---

## Exemplo 8

**Frase:**   
**Prompt:** `6480 23392 2629 5464 7064 8476`  
**Resposta correta:** ID `26821`  
**Resultado:** a resposta correta apareceu na posição **1**.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 26821 | team | 0.77734375 | ✅ |
| 2 | 15262 | team | 0.11914062 |  |
| 3 | 15264 | team | 0.06396484 |  |
| 4 | 37544 | parliamentary group | 0.01611328 |  |
| 5 | 12333 | organization | 0.01257324 |  |
| 6 | 29919 | sports club | 0.00976562 |  |
| 7 | 2704 | city | 0.00009584 |  |
| 8 | 37651 | Ireland | 0.00006199 |  |
| 9 | 22051 | Poland | 0.00004530 |  |
| 10 | 24667 | social intervention team | 0.00004005 |  |
| 11 | 7137 | soccer | 0.00001895 |  |
| 12 | 28985 | Poland | 0.00001222 |  |
| 13 | 34449 | water polo player | 0.00001079 |  |
| 14 | 37406 | write the title | 0.00001013 |  |
| 15 | 6161 | gold | 0.00001013 |  |
| 16 | 4726 | first | 0.00001013 |  |
| 17 | 8476 | the | 0.00000787 |  |
| 18 | 11709 | at the | 0.00000694 |  |
| 19 | 36367 | french class | 0.00000614 |  |
| 20 | 39791 | basketball team | 0.00000510 |  |
| 21 | 7074 | of | 0.00000477 |  |
| 22 | 37779 | somebody | 0.00000423 |  |
| 23 | 28545 | club | 0.00000396 |  |
| 24 | 21844 | Scotland | 0.00000373 |  |
| 25 | 21941 | Ireland | 0.00000328 |  |

---

## Exemplo 9

**Frase:**   
**Prompt:** `8476 24378 6048`  
**Resposta correta:** ID `36297`  
**Resultado:** a resposta correta apareceu na posição **13**.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 2704 | city | 0.67968750 |  |
| 2 | 7034 | in | 0.08642578 |  |
| 3 | 11399 | and | 0.05590820 |  |
| 4 | 8476 | the | 0.05590820 |  |
| 5 | 11709 | at the | 0.04345703 |  |
| 6 | 5526 | no | 0.01599121 |  |
| 7 | 12333 | organization | 0.01245117 |  |
| 8 | 11351 | that | 0.00970459 |  |
| 9 | 2627 | one | 0.00756836 |  |
| 10 | 7074 | of | 0.00457764 |  |
| 11 | 12313 | how | 0.00430298 |  |
| 12 | 5800 | no | 0.00430298 |  |
| 13 | 36297 | monument | 0.00277710 | ✅ |
| 14 | 7041 | to | 0.00139618 |  |
| 15 | 5596 | all | 0.00131226 |  |
| 16 | 5525 | no | 0.00131226 |  |
| 17 | 8474 | a | 0.00123596 |  |
| 18 | 6906 | that one | 0.00099182 |  |
| 19 | 23398 | interculturality | 0.00099182 |  |
| 20 | 7091 | that | 0.00099182 |  |
| 21 | 2635 | 9 | 0.00093079 |  |
| 22 | 8173 | worldwide | 0.00074768 |  |
| 23 | 6481 | he | 0.00045395 |  |
| 24 | 35689 | 9 | 0.00034332 |  |
| 25 | 7212 | by | 0.00034332 |  |

---

## Exemplo 10

**Frase:**   
**Prompt:** `8474 2456 7074 3082`  
**Resposta correta:** ID `38570`  
**Resultado:** a resposta correta apareceu na posição **21**.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 36480 | be | 0.90625000 |  |
| 2 | 2627 | one | 0.03100586 |  |
| 3 | 36935 | half body shot | 0.02416992 |  |
| 4 | 5581 | be | 0.01141357 |  |
| 5 | 36392 | be | 0.00787354 |  |
| 6 | 7041 | to | 0.00650024 |  |
| 7 | 8476 | the | 0.00555420 |  |
| 8 | 7074 | of | 0.00280762 |  |
| 9 | 35949 | be able to | 0.00120544 |  |
| 10 | 5466 | be | 0.00020313 |  |
| 11 | 5858 | be | 0.00020313 |  |
| 12 | 5465 | be | 0.00020313 |  |
| 13 | 35884 | secret ballot | 0.00015831 |  |
| 14 | 32816 | wish for | 0.00012684 |  |
| 15 | 2457 | teacher | 0.00009012 |  |
| 16 | 8477 | the | 0.00007677 |  |
| 17 | 36377 | be accountable to | 0.00004959 |  |
| 18 | 16719 | retire | 0.00004816 |  |
| 19 | 6906 | that one | 0.00004601 |  |
| 20 | 7194 | for | 0.00003529 |  |
| 21 | 38570 | what is your name? | 0.00003362 | ✅ |
| 22 | 24711 | be nice | 0.00003219 |  |
| 23 | 11399 | and | 0.00003016 |  |
| 24 | 26818 | put together | 0.00002873 |  |
| 25 | 6624 | work | 0.00002038 |  |

---

## Exemplo 11

**Frase:**   
**Prompt:** `6480 21531 7212`  
**Resposta correta:** ID `12333`  
**Resultado:** a resposta correta apareceu na posição **11**.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 8476 | the | 0.73828125 |  |
| 2 | 8477 | the | 0.24023438 |  |
| 3 | 12272 | his | 0.00497437 |  |
| 4 | 12276 | his | 0.00341797 |  |
| 5 | 8121 | France | 0.00267029 |  |
| 6 | 36935 | half body shot | 0.00267029 |  |
| 7 | 8474 | a | 0.00183105 |  |
| 8 | 11399 | and | 0.00142670 |  |
| 9 | 7074 | of | 0.00125885 |  |
| 10 | 2630 | 4 | 0.00086594 |  |
| 11 | 12333 | organization | 0.00067520 | ✅ |
| 12 | 2627 | one | 0.00040817 |  |
| 13 | 11709 | at the | 0.00023270 |  |
| 14 | 2628 | 2 | 0.00010347 |  |
| 15 | 7248 | 7 | 0.00009108 |  |
| 16 | 37779 | somebody | 0.00008583 |  |
| 17 | 32394 | uneven | 0.00008583 |  |
| 18 | 19570 | 1/2 | 0.00005889 |  |
| 19 | 2632 | 6 | 0.00005889 |  |
| 20 | 3086 | postal service | 0.00004578 |  |
| 21 | 37544 | parliamentary group | 0.00004053 |  |
| 22 | 7041 | to | 0.00003791 |  |
| 23 | 6979 | 5 | 0.00003362 |  |
| 24 | 30014 | The Earth | 0.00003362 |  |
| 25 | 5375 | there | 0.00002956 |  |

---

## Exemplo 12

**Frase:**   
**Prompt:** `6632 37721 6190 11712`  
**Resposta correta:** ID `9058`  
**Resultado:** a resposta correta apareceu na posição **1**.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 9058 | shopping | 1.00000000 | ✅ |
| 2 | 11211 | shopping centre | 0.00000450 |  |
| 3 | 26748 | shopping basket | 0.00000200 |  |
| 4 | 7074 | of | 0.00000156 |  |
| 5 | 34781 | make fun of | 0.00000008 |  |
| 6 | 15551 | shopping centre | 0.00000003 |  |
| 7 | 26705 | make fun of | 0.00000002 |  |
| 8 | 29947 | game of yo-yo | 0.00000002 |  |
| 9 | 26704 | make fun | 0.00000001 |  |
| 10 | 9067 | llama | 0.00000001 |  |
| 11 | 9185 | Point of Sale Terminal | 0.00000001 |  |
| 12 | 6962 | shopping troley | 0.00000001 |  |
| 13 | 8673 | No drinking | 0.00000001 |  |
| 14 | 32800 | festivities | 0.00000000 |  |
| 15 | 5992 | shop window | 0.00000000 |  |
| 16 | 6642 | Poetry and Art's books | 0.00000000 |  |
| 17 | 36387 | elf | 0.00000000 |  |
| 18 | 7308 | yolk | 0.00000000 |  |
| 19 | 8674 | No eating | 0.00000000 |  |
| 20 | 26901 | footsteps | 0.00000000 |  |
| 21 | 23859 | cape | 0.00000000 |  |
| 22 | 4768 | glass of water | 0.00000000 |  |
| 23 | 2479 | mosquito | 0.00000000 |  |
| 24 | 7216 | dessert | 0.00000000 |  |
| 25 | 4882 | sweet shop | 0.00000000 |  |

---

## Exemplo 13

**Frase:**   
**Prompt:** `6480 5581 2628 17056 9902 32806`  
**Resposta correta:** ID `6939`  
**Resultado:** a resposta correta não apareceu no top-25.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 11351 | that | 0.44140625 |  |
| 2 | 11709 | at the | 0.10400391 |  |
| 3 | 3423 | equal sign | 0.07714844 |  |
| 4 | 36778 | ly | 0.07031250 |  |
| 5 | 7091 | that | 0.05395508 |  |
| 6 | 8477 | the | 0.03710938 |  |
| 7 | 24509 | CD player | 0.02624512 |  |
| 8 | 3043 | u | 0.01916504 |  |
| 9 | 11399 | and | 0.01586914 |  |
| 10 | 8476 | the | 0.01202393 |  |
| 11 | 3041 | s | 0.00964355 |  |
| 12 | 6642 | Poetry and Art's books | 0.00909424 |  |
| 13 | 34449 | water polo player | 0.00878906 |  |
| 14 | 7074 | of | 0.00823975 |  |
| 15 | 8611 | rugby player | 0.00823975 |  |
| 16 | 27292 | basketball player | 0.00665283 |  |
| 17 | 2704 | city | 0.00567627 |  |
| 18 | 5080 | ice-hockey player | 0.00567627 |  |
| 19 | 6906 | that one | 0.00534058 |  |
| 20 | 11712 | of the | 0.00534058 |  |
| 21 | 8609 | baseball player | 0.00469971 |  |
| 22 | 5079 | handball player | 0.00402832 |  |
| 23 | 24424 | keyboard player | 0.00366211 |  |
| 24 | 34920 | basketball player | 0.00355530 |  |
| 25 | 8115 | and that is the end of the story | 0.00303650 |  |

---

## Exemplo 14

**Frase:**   
**Prompt:** `12272 7795 7074 4658`  
**Resposta correta:** ID `7294`  
**Resultado:** a resposta correta não apareceu no top-25.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 7074 | of | 0.40429688 |  |
| 2 | 14258 | wig | 0.09179688 |  |
| 3 | 38570 | what is your name? | 0.08886719 |  |
| 4 | 11399 | and | 0.08105469 |  |
| 5 | 11709 | at the | 0.06103516 |  |
| 6 | 11475 | take part | 0.02758789 |  |
| 7 | 24731 | how many? | 0.02465820 |  |
| 8 | 36480 | be | 0.01696777 |  |
| 9 | 6481 | he | 0.01373291 |  |
| 10 | 2704 | city | 0.01214600 |  |
| 11 | 6531 | have an idea | 0.01123047 |  |
| 12 | 32792 | Which is the difference? | 0.00976562 |  |
| 13 | 21371 | bet | 0.00598145 |  |
| 14 | 8476 | the | 0.00500488 |  |
| 15 | 34781 | make fun of | 0.00485229 |  |
| 16 | 24177 | mosque | 0.00402832 |  |
| 17 | 14660 | honour | 0.00396729 |  |
| 18 | 38076 | pit | 0.00393677 |  |
| 19 | 12274 | theirs | 0.00378418 |  |
| 20 | 5925 | whisk | 0.00349426 |  |
| 21 | 26427 | mummy | 0.00343323 |  |
| 22 | 5368 | wing | 0.00320435 |  |
| 23 | 12256 | war | 0.00320435 |  |
| 24 | 2358 | croissant | 0.00315857 |  |
| 25 | 37708 | recommend | 0.00309753 |  |

---

## Exemplo 15

**Frase:**   
**Prompt:** `7028 7041 8476 7074 8476 8693`  
**Resposta correta:** ID `37544`  
**Resultado:** a resposta correta apareceu na posição **8**.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 7074 | of | 0.47070312 |  |
| 2 | 6152 | indigenous person of India | 0.11865234 |  |
| 3 | 12333 | organization | 0.10498047 |  |
| 4 | 25165 | India | 0.07226562 |  |
| 5 | 28978 | Switzerland | 0.06347656 |  |
| 6 | 5958 | indigenous person of Asia | 0.03393555 |  |
| 7 | 6282 | indigenous person of Africa | 0.02343750 |  |
| 8 | 37544 | parliamentary group | 0.02062988 | ✅ |
| 9 | 28976 | Russia | 0.00714111 |  |
| 10 | 27603 | Burma | 0.00714111 |  |
| 11 | 8144 | Japan | 0.00671387 |  |
| 12 | 36935 | half body shot | 0.00671387 |  |
| 13 | 8112 | United States of America | 0.00592041 |  |
| 14 | 36367 | french class | 0.00460815 |  |
| 15 | 2609 | cow | 0.00405884 |  |
| 16 | 11709 | at the | 0.00381470 |  |
| 17 | 5377 | America | 0.00358582 |  |
| 18 | 21393 | Australia | 0.00358582 |  |
| 19 | 8178 | North | 0.00315857 |  |
| 20 | 37740 | german class | 0.00297546 |  |
| 21 | 8206 | United Kingdom | 0.00231934 |  |
| 22 | 2704 | city | 0.00204468 |  |
| 23 | 29586 | Slovenia | 0.00204468 |  |
| 24 | 37647 | China | 0.00192261 |  |
| 25 | 6153 | indigenous person of the Americas | 0.00192261 |  |

---

## Exemplo 16

**Frase:**   
**Prompt:** `7028 35545 7041 8476 11246 7074 32394`  
**Resposta correta:** ID `26740`  
**Resultado:** a resposta correta não apareceu no top-25.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 2704 | city | 0.86718750 |  |
| 2 | 37544 | parliamentary group | 0.12109375 |  |
| 3 | 11246 | army | 0.00328064 |  |
| 4 | 37851 | stamina | 0.00164795 |  |
| 5 | 35973 | army | 0.00066376 |  |
| 6 | 14755 | La Coruña | 0.00054932 |  |
| 7 | 10269 | artery | 0.00053787 |  |
| 8 | 5520 | Moncayo | 0.00049591 |  |
| 9 | 38202 | clear all button | 0.00045204 |  |
| 10 | 3428 | ca-ca | 0.00037384 |  |
| 11 | 28069 | Turkmenistan | 0.00021744 |  |
| 12 | 12315 | comply | 0.00016212 |  |
| 13 | 36697 | criticise | 0.00013733 |  |
| 14 | 36287 | maize | 0.00012684 |  |
| 15 | 9069 | reindeer | 0.00009918 |  |
| 16 | 6531 | have an idea | 0.00009012 |  |
| 17 | 35401 | social sciences | 0.00007391 |  |
| 18 | 36897 | Burgos' Cathedral | 0.00006628 |  |
| 19 | 4728 | cigar | 0.00006104 |  |
| 20 | 32452 | Saint John's day | 0.00005817 |  |
| 21 | 38015 | sedentary | 0.00005674 |  |
| 22 | 3040 | r | 0.00005412 |  |
| 23 | 30023 | embassy | 0.00004983 |  |
| 24 | 8008 | activity | 0.00004864 |  |
| 25 | 7809 | sardina | 0.00004816 |  |

---

## Exemplo 17

**Frase:**   
**Prompt:** `5525 9839 36935 32761 5800 11164 7074`  
**Resposta correta:** ID `7218`  
**Resultado:** a resposta correta não apareceu no top-25.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 30604 | play sports | 0.23925781 |  |
| 2 | 8476 | the | 0.22558594 |  |
| 3 | 3089 | dice game | 0.11572266 |  |
| 4 | 31670 | It | 0.05834961 |  |
| 5 | 27306 | Monaco | 0.04858398 |  |
| 6 | 35949 | be able to | 0.04223633 |  |
| 7 | 38413 | eat with the hand | 0.02270508 |  |
| 8 | 35884 | secret ballot | 0.02160645 |  |
| 9 | 25752 | you got it! | 0.02050781 |  |
| 10 | 36935 | half body shot | 0.01464844 |  |
| 11 | 37900 | slip the sand | 0.00769043 |  |
| 12 | 11163 | guess | 0.00650024 |  |
| 13 | 28067 | Sri Lanka | 0.00546265 |  |
| 14 | 7180 | I do not know | 0.00506592 |  |
| 15 | 8555 | turn on the monitor | 0.00488281 |  |
| 16 | 30636 | look down on | 0.00445557 |  |
| 17 | 36151 | fortune-teller | 0.00439453 |  |
| 18 | 12274 | theirs | 0.00405884 |  |
| 19 | 15497 | put the bra on | 0.00347900 |  |
| 20 | 30014 | The Earth | 0.00326538 |  |
| 21 | 31004 | look down on | 0.00306702 |  |
| 22 | 3152 | S | 0.00288391 |  |
| 23 | 6481 | he | 0.00288391 |  |
| 24 | 8477 | the | 0.00274658 |  |
| 25 | 23392 | play | 0.00270081 |  |

---

## Exemplo 18

**Frase:**   
**Prompt:** `7028 8243 2704 7041`  
**Resposta correta:** ID `2704`  
**Resultado:** a resposta correta apareceu na posição **1**.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 2704 | city | 0.95703125 | ✅ |
| 2 | 6906 | that one | 0.03271484 |  |
| 3 | 2627 | one | 0.00185394 |  |
| 4 | 14771 | Burgos | 0.00185394 |  |
| 5 | 7074 | of | 0.00099182 |  |
| 6 | 8476 | the | 0.00099182 |  |
| 7 | 7091 | that | 0.00068283 |  |
| 8 | 14755 | La Coruña | 0.00036430 |  |
| 9 | 23769 | Atomium | 0.00028419 |  |
| 10 | 10271 | Eiffel Tower | 0.00028419 |  |
| 11 | 23957 | Rome Coliseum | 0.00022125 |  |
| 12 | 5367 | next to | 0.00007200 |  |
| 13 | 11351 | that | 0.00007200 |  |
| 14 | 21445 | Big Ben | 0.00005579 |  |
| 15 | 32452 | Saint John's day | 0.00004625 |  |
| 16 | 34165 | rook | 0.00004625 |  |
| 17 | 21854 | Statue of Liberty | 0.00004625 |  |
| 18 | 30196 | feel | 0.00004363 |  |
| 19 | 36897 | Burgos' Cathedral | 0.00002813 |  |
| 20 | 5963 | city | 0.00002480 |  |
| 21 | 14772 | Leon | 0.00002480 |  |
| 22 | 11709 | at the | 0.00002325 |  |
| 23 | 29730 | Bahamas | 0.00002050 |  |
| 24 | 14832 | Barcelona | 0.00001931 |  |
| 25 | 7212 | by | 0.00001413 |  |

---

## Exemplo 19

**Frase:**   
**Prompt:** `6480 36480 34784 7074 8476`  
**Resposta correta:** ID `36367`  
**Resultado:** a resposta correta apareceu na posição **3**.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 30387 | cinema | 0.27539062 |  |
| 2 | 12333 | organization | 0.16699219 |  |
| 3 | 36367 | french class | 0.14746094 | ✅ |
| 4 | 36345 | community association | 0.14746094 |  |
| 5 | 16129 | department | 0.07861328 |  |
| 6 | 37544 | parliamentary group | 0.04760742 |  |
| 7 | 2704 | city | 0.01367188 |  |
| 8 | 21906 | government | 0.01062012 |  |
| 9 | 36794 | profile photograph | 0.00939941 |  |
| 10 | 7064 | with | 0.00830078 |  |
| 11 | 8693 | society | 0.00646973 |  |
| 12 | 36935 | half body shot | 0.00570679 |  |
| 13 | 36365 | philosophy class | 0.00570679 |  |
| 14 | 7074 | of | 0.00570679 |  |
| 15 | 4602 | cinema | 0.00442505 |  |
| 16 | 37740 | german class | 0.00390625 |  |
| 17 | 3060 | town hall | 0.00390625 |  |
| 18 | 34161 | academy | 0.00346375 |  |
| 19 | 2845 | newspaper | 0.00305176 |  |
| 20 | 10271 | Eiffel Tower | 0.00268555 |  |
| 21 | 36897 | Burgos' Cathedral | 0.00268555 |  |
| 22 | 4658 | large | 0.00238037 |  |
| 23 | 9815 | classroom | 0.00238037 |  |
| 24 | 15537 | University | 0.00209045 |  |
| 25 | 8121 | France | 0.00209045 |  |

---

## Exemplo 20

**Frase:**   
**Prompt:** `9839 6480 38428 7064 8477`  
**Resposta correta:** ID `36935`  
**Resultado:** a resposta correta apareceu na posição **11**.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 12333 | organization | 0.98046875 |  |
| 2 | 7074 | of | 0.01086426 |  |
| 3 | 37544 | parliamentary group | 0.00659180 |  |
| 4 | 32564 | hope | 0.00101471 |  |
| 5 | 8112 | United States of America | 0.00069427 |  |
| 6 | 36367 | french class | 0.00008297 |  |
| 7 | 2704 | city | 0.00005031 |  |
| 8 | 11351 | that | 0.00004745 |  |
| 9 | 6642 | Poetry and Art's books | 0.00004458 |  |
| 10 | 2449 | lion | 0.00003934 |  |
| 11 | 36935 | half body shot | 0.00003457 | ✅ |
| 12 | 32818 | bottom | 0.00003266 |  |
| 13 | 7784 | news | 0.00002873 |  |
| 14 | 6531 | have an idea | 0.00002873 |  |
| 15 | 25187 | lion | 0.00002384 |  |
| 16 | 7091 | that | 0.00002098 |  |
| 17 | 37752 | first time | 0.00001740 |  |
| 18 | 11247 | rights | 0.00001359 |  |
| 19 | 5438 | in front | 0.00001276 |  |
| 20 | 9185 | Point of Sale Terminal | 0.00001276 |  |
| 21 | 21854 | Statue of Liberty | 0.00001276 |  |
| 22 | 6152 | indigenous person of India | 0.00000995 |  |
| 23 | 37406 | write the title | 0.00000823 |  |
| 24 | 11399 | and | 0.00000530 |  |
| 25 | 6153 | indigenous person of the Americas | 0.00000530 |  |

---

## Exemplo 21

**Frase:**   
**Prompt:** `12272 37138 36480`  
**Resposta correta:** ID `3340`  
**Resultado:** a resposta correta apareceu na posição **10**.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 2662 | white | 0.43164062 |  |
| 2 | 2648 | yellow | 0.12402344 |  |
| 3 | 11399 | and | 0.10937500 |  |
| 4 | 2888 | orange | 0.08496094 |  |
| 5 | 7074 | of | 0.03540039 |  |
| 6 | 2923 | brown | 0.03540039 |  |
| 7 | 4887 | green | 0.03125000 |  |
| 8 | 4869 | blue | 0.01904297 |  |
| 9 | 9093 | beige | 0.01904297 |  |
| 10 | 3340 | grey | 0.01306152 | ✅ |
| 11 | 4886 | green | 0.00897217 |  |
| 12 | 8184 | red-haired | 0.00793457 |  |
| 13 | 9094 | coffee-coloured | 0.00698853 |  |
| 14 | 26993 | dark | 0.00616455 |  |
| 15 | 25744 | mustard-colored | 0.00616455 |  |
| 16 | 2886 | black | 0.00616455 |  |
| 17 | 3355 | blue | 0.00616455 |  |
| 18 | 26266 | dark | 0.00373840 |  |
| 19 | 2808 | red | 0.00329590 |  |
| 20 | 11317 | or | 0.00291443 |  |
| 21 | 11332 | hairy | 0.00291443 |  |
| 22 | 25708 | very | 0.00256348 |  |
| 23 | 4667 | equal | 0.00227356 |  |
| 24 | 32392 | equal | 0.00199890 |  |
| 25 | 4603 | circle | 0.00177002 |  |

---

## Exemplo 22

**Frase:**   
**Prompt:** `6480 32250 7074 2627 38848 7074`  
**Resposta correta:** ID `2972`  
**Resultado:** a resposta correta não apareceu no top-25.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 2696 | brain | 0.93750000 |  |
| 2 | 2853 | chest | 0.02502441 |  |
| 3 | 8476 | the | 0.01336670 |  |
| 4 | 27365 | face to face | 0.00762939 |  |
| 5 | 19548 | 1-2-3 | 0.00717163 |  |
| 6 | 37164 | chest | 0.00317383 |  |
| 7 | 33050 | face-to-face work | 0.00247192 |  |
| 8 | 11709 | at the | 0.00109863 |  |
| 9 | 19570 | 1/2 | 0.00033569 |  |
| 10 | 7041 | to | 0.00016880 |  |
| 11 | 31773 | urinary tract infection | 0.00013161 |  |
| 12 | 16349 | Alzheimer's disease | 0.00007486 |  |
| 13 | 5367 | next to | 0.00007010 |  |
| 14 | 37768 | ring-a-ring-a-roses | 0.00006819 |  |
| 15 | 38934 | thump the chest | 0.00006390 |  |
| 16 | 6198 | X-rays | 0.00006390 |  |
| 17 | 5436 | 1/4 | 0.00004268 |  |
| 18 | 11241 | pain in the chest | 0.00003886 |  |
| 19 | 2628 | 2 | 0.00003314 |  |
| 20 | 30108 | x-ray | 0.00002837 |  |
| 21 | 30109 | x-ray | 0.00002837 |  |
| 22 | 2328 | cereals | 0.00002754 |  |
| 23 | 8477 | the | 0.00002754 |  |
| 24 | 36426 | the finger | 0.00002277 |  |
| 25 | 36387 | elf | 0.00002277 |  |

---

## Exemplo 23

**Frase:**   
**Prompt:** `5596 9839 8476 6190`  
**Resposta correta:** ID `5936`  
**Resultado:** a resposta correta não apareceu no top-25.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 32065 | victim | 0.32031250 |  |
| 2 | 5451 | above | 0.24902344 |  |
| 3 | 29839 | nothing | 0.17089844 |  |
| 4 | 7034 | in | 0.09130859 |  |
| 5 | 7074 | of | 0.07128906 |  |
| 6 | 7041 | to | 0.04907227 |  |
| 7 | 7212 | by | 0.01239014 |  |
| 8 | 8476 | the | 0.01092529 |  |
| 9 | 7194 | for | 0.00704956 |  |
| 10 | 11399 | and | 0.00515747 |  |
| 11 | 11709 | at the | 0.00355530 |  |
| 12 | 8477 | the | 0.00276184 |  |
| 13 | 32522 | nothing | 0.00202942 |  |
| 14 | 6481 | he | 0.00045204 |  |
| 15 | 38617 | stages of life | 0.00045204 |  |
| 16 | 12313 | how | 0.00035095 |  |
| 17 | 36480 | be | 0.00022697 |  |
| 18 | 38753 | toggle | 0.00022697 |  |
| 19 | 11351 | that | 0.00020027 |  |
| 20 | 17322 | always | 0.00020027 |  |
| 21 | 7217 | question | 0.00014687 |  |
| 22 | 36989 | echo | 0.00014687 |  |
| 23 | 32360 | nostalgic | 0.00013733 |  |
| 24 | 32592 | expression | 0.00013733 |  |
| 25 | 11712 | of the | 0.00010729 |  |

---

## Exemplo 24

**Frase:**   
**Prompt:** `8477 7074 6018 7074 32090 7041`  
**Resposta correta:** ID `10337`  
**Resultado:** a resposta correta não apareceu no top-25.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 36514 | cupola | 0.50781250 |  |
| 2 | 35505 | indicator | 0.03540039 |  |
| 3 | 3428 | ca-ca | 0.03271484 |  |
| 4 | 19548 | 1-2-3 | 0.03198242 |  |
| 5 | 2962 | syrup | 0.02990723 |  |
| 6 | 29965 | chasuble | 0.02526855 |  |
| 7 | 8184 | red-haired | 0.02355957 |  |
| 8 | 3218 | point sign | 0.01696777 |  |
| 9 | 5478 | giants | 0.01599121 |  |
| 10 | 35117 | cents | 0.01586914 |  |
| 11 | 8496 | low-cal | 0.01226807 |  |
| 12 | 34343 | slap | 0.01049805 |  |
| 13 | 24400 | Shiva | 0.00799561 |  |
| 14 | 31316 | Port-a-Cath | 0.00701904 |  |
| 15 | 37768 | ring-a-ring-a-roses | 0.00695801 |  |
| 16 | 16345 | double | 0.00634766 |  |
| 17 | 21874 | float | 0.00555420 |  |
| 18 | 9185 | Point of Sale Terminal | 0.00537109 |  |
| 19 | 32876 | head pointer | 0.00534058 |  |
| 20 | 35755 | arugula | 0.00439453 |  |
| 21 | 8476 | the | 0.00411987 |  |
| 22 | 2338 | Coca-Cola | 0.00402832 |  |
| 23 | 6945 | bubbles | 0.00364685 |  |
| 24 | 29947 | game of yo-yo | 0.00334167 |  |
| 25 | 3090 | darts | 0.00321960 |  |

---

## Exemplo 25

**Frase:**   
**Prompt:** `8476 37544 7074 2704 32757 5451 12272`  
**Resposta correta:** ID `8179`  
**Resultado:** a resposta correta não apareceu no top-25.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 35431 | country | 0.53515625 |  |
| 2 | 6186 | sports centre | 0.39062500 |  |
| 3 | 9819 | place | 0.03881836 |  |
| 4 | 29919 | sports club | 0.01721191 |  |
| 5 | 9203 | left | 0.00509644 |  |
| 6 | 32598 | place | 0.00299072 |  |
| 7 | 5424 | centre | 0.00124359 |  |
| 8 | 36249 | historic district | 0.00109863 |  |
| 9 | 37544 | parliamentary group | 0.00091171 |  |
| 10 | 6188 | south pole | 0.00088501 |  |
| 11 | 38586 | village | 0.00085449 |  |
| 12 | 6039 | photographic album | 0.00051880 |  |
| 13 | 6187 | North Pole | 0.00043106 |  |
| 14 | 11211 | shopping centre | 0.00043106 |  |
| 15 | 8228 | South | 0.00031471 |  |
| 16 | 27399 | connect to internet | 0.00029564 |  |
| 17 | 15748 | centre | 0.00025368 |  |
| 18 | 4672 | left | 0.00019073 |  |
| 19 | 29845 | social centre | 0.00013161 |  |
| 20 | 8178 | North | 0.00013161 |  |
| 21 | 24204 | building site | 0.00011396 |  |
| 22 | 27093 | piece | 0.00010729 |  |
| 23 | 32848 | ante meridiem | 0.00010538 |  |
| 24 | 3118 | church | 0.00009918 |  |
| 25 | 32504 | sports centre | 0.00009012 |  |

---

## Exemplo 26

**Frase:**   
**Prompt:** `8477 16087 15816 7212`  
**Resposta correta:** ID `36935`  
**Resultado:** a resposta correta apareceu na posição **19**.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 8476 | the | 0.86718750 |  |
| 2 | 8477 | the | 0.12500000 |  |
| 3 | 2627 | one | 0.00276184 |  |
| 4 | 7074 | of | 0.00157166 |  |
| 5 | 23371 | success | 0.00035095 |  |
| 6 | 11709 | at the | 0.00035095 |  |
| 7 | 11712 | of the | 0.00031090 |  |
| 8 | 12274 | theirs | 0.00025749 |  |
| 9 | 8474 | a | 0.00022030 |  |
| 10 | 12272 | his | 0.00020027 |  |
| 11 | 38564 | success | 0.00010729 |  |
| 12 | 12278 | theirs | 0.00007343 |  |
| 13 | 7041 | to | 0.00006294 |  |
| 14 | 11599 | success | 0.00004601 |  |
| 15 | 2628 | 2 | 0.00002623 |  |
| 16 | 5374 | some | 0.00001621 |  |
| 17 | 37544 | parliamentary group | 0.00001258 |  |
| 18 | 6906 | that one | 0.00001240 |  |
| 19 | 36935 | half body shot | 0.00001168 | ✅ |
| 20 | 12276 | his | 0.00000906 |  |
| 21 | 5516 | two quarters | 0.00000569 |  |
| 22 | 2629 | 3 | 0.00000486 |  |
| 23 | 29482 | three o'clock | 0.00000282 |  |
| 24 | 2632 | 6 | 0.00000216 |  |
| 25 | 19570 | 1/2 | 0.00000171 |  |

---

## Exemplo 27

**Frase:**   
**Prompt:** `7028 8207 7034 8474 2729 7074`  
**Resposta correta:** ID `2704`  
**Resultado:** a resposta correta apareceu na posição **2**.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 6903 | year | 0.97265625 |  |
| 2 | 2704 | city | 0.01574707 | ✅ |
| 3 | 8476 | the | 0.01080322 |  |
| 4 | 27608 | cellar | 0.00069046 |  |
| 5 | 7074 | of | 0.00028801 |  |
| 6 | 5367 | next to | 0.00019741 |  |
| 7 | 2729 | cave | 0.00012779 |  |
| 8 | 37779 | somebody | 0.00010586 |  |
| 9 | 7041 | to | 0.00008774 |  |
| 10 | 15996 | larder | 0.00003219 |  |
| 11 | 5520 | Moncayo | 0.00002217 |  |
| 12 | 2683 | countryside | 0.00002086 |  |
| 13 | 4728 | cigar | 0.00001729 |  |
| 14 | 39732 | next year | 0.00000465 |  |
| 15 | 10269 | artery | 0.00000411 |  |
| 16 | 37544 | parliamentary group | 0.00000319 |  |
| 17 | 6241 | neighbours | 0.00000319 |  |
| 18 | 36935 | half body shot | 0.00000282 |  |
| 19 | 35069 | God | 0.00000219 |  |
| 20 | 2630 | 4 | 0.00000194 |  |
| 21 | 32452 | Saint John's day | 0.00000194 |  |
| 22 | 26529 | neighbours | 0.00000194 |  |
| 23 | 8261 | New Year's Eve | 0.00000151 |  |
| 24 | 36897 | Burgos' Cathedral | 0.00000151 |  |
| 25 | 6979 | 5 | 0.00000142 |  |

---

## Exemplo 28

**Frase:**   
**Prompt:** `9839 36935 7041 8476 11399 12276 2876`  
**Resposta correta:** ID `31314`  
**Resultado:** a resposta correta não apareceu no top-25.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 10366 | what is that? | 0.09667969 |  |
| 2 | 11712 | of the | 0.08593750 |  |
| 3 | 21375 | lean on | 0.07763672 |  |
| 4 | 2637 | open the box | 0.06982422 |  |
| 5 | 6642 | Poetry and Art's books | 0.06127930 |  |
| 6 | 7264 | put the top on | 0.04882812 |  |
| 7 | 7803 | Who's calling? | 0.04248047 |  |
| 8 | 34367 | get up from the floor | 0.03906250 |  |
| 9 | 9072 | pick up a package | 0.03198242 |  |
| 10 | 37821 | peeler | 0.03125000 |  |
| 11 | 15497 | put the bra on | 0.02941895 |  |
| 12 | 7074 | of | 0.02209473 |  |
| 13 | 11611 | what are you saying? | 0.02099609 |  |
| 14 | 32394 | uneven | 0.01611328 |  |
| 15 | 3418 | ? | 0.01483154 |  |
| 16 | 32386 | uneven | 0.01458740 |  |
| 17 | 30636 | look down on | 0.01214600 |  |
| 18 | 5096 | put on cream | 0.01159668 |  |
| 19 | 38570 | what is your name? | 0.01043701 |  |
| 20 | 11617 | can you see it? | 0.00939941 |  |
| 21 | 8555 | turn on the monitor | 0.00927734 |  |
| 22 | 8476 | the | 0.00878906 |  |
| 23 | 31004 | look down on | 0.00872803 |  |
| 24 | 7223 | what's the weather like? | 0.00781250 |  |
| 25 | 2746 | listen to music | 0.00762939 |  |

---

## Exemplo 29

**Frase:**   
**Prompt:** `9839 6480 7074`  
**Resposta correta:** ID `35581`  
**Resultado:** a resposta correta não apareceu no top-25.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 7168 | many | 0.26953125 |  |
| 2 | 8476 | the | 0.16308594 |  |
| 3 | 25656 | movement | 0.06005859 |  |
| 4 | 37708 | recommend | 0.04125977 |  |
| 5 | 5512 | less | 0.04125977 |  |
| 6 | 35931 | July | 0.03222656 |  |
| 7 | 2704 | city | 0.02844238 |  |
| 8 | 8474 | a | 0.02502441 |  |
| 9 | 12272 | his | 0.02502441 |  |
| 10 | 8147 | law court | 0.02209473 |  |
| 11 | 12276 | his | 0.01519775 |  |
| 12 | 24742 | find | 0.01519775 |  |
| 13 | 7064 | with | 0.01519775 |  |
| 14 | 11473 | law | 0.01519775 |  |
| 15 | 7074 | of | 0.01184082 |  |
| 16 | 2628 | 2 | 0.01043701 |  |
| 17 | 36480 | be | 0.00921631 |  |
| 18 | 35945 | February | 0.00921631 |  |
| 19 | 24741 | find | 0.00921631 |  |
| 20 | 12313 | how | 0.00921631 |  |
| 21 | 35923 | January | 0.00811768 |  |
| 22 | 24773 | find | 0.00811768 |  |
| 23 | 7041 | to | 0.00717163 |  |
| 24 | 35935 | April | 0.00558472 |  |
| 25 | 35927 | November | 0.00436401 |  |

---

## Exemplo 30

**Frase:**   
**Prompt:** `7032 6190`  
**Resposta correta:** ID `7074`  
**Resultado:** a resposta correta apareceu na posição **2**.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 27093 | piece | 0.34375000 |  |
| 2 | 7074 | of | 0.12695312 | ✅ |
| 3 | 5596 | all | 0.09863281 |  |
| 4 | 27091 | all | 0.05981445 |  |
| 5 | 2627 | one | 0.05273438 |  |
| 6 | 8474 | a | 0.04101562 |  |
| 7 | 12274 | theirs | 0.03637695 |  |
| 8 | 7212 | by | 0.02490234 |  |
| 9 | 5397 | well | 0.01708984 |  |
| 10 | 12278 | theirs | 0.01708984 |  |
| 11 | 8476 | the | 0.01513672 |  |
| 12 | 8243 | join | 0.01513672 |  |
| 13 | 5526 | no | 0.01513672 |  |
| 14 | 11377 | but | 0.01336670 |  |
| 15 | 11351 | that | 0.00915527 |  |
| 16 | 32438 | all | 0.00915527 |  |
| 17 | 7041 | to | 0.00915527 |  |
| 18 | 8477 | the | 0.00714111 |  |
| 19 | 3423 | equal sign | 0.00631714 |  |
| 20 | 7194 | for | 0.00555420 |  |
| 21 | 11709 | at the | 0.00491333 |  |
| 22 | 11712 | of the | 0.00433350 |  |
| 23 | 5374 | some | 0.00382996 |  |
| 24 | 8456 | know | 0.00297546 |  |
| 25 | 5508 | more | 0.00262451 |  |

---

## Exemplo 31

**Frase:**   
**Prompt:** `7194 8476`  
**Resposta correta:** ID `2704`  
**Resultado:** a resposta correta não apareceu no top-25.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 37752 | first time | 0.07861328 |  |
| 2 | 12295 | cause | 0.06933594 |  |
| 3 | 8476 | the | 0.06127930 |  |
| 4 | 7074 | of | 0.05419922 |  |
| 5 | 4665 | man | 0.04785156 |  |
| 6 | 38004 | essence | 0.03295898 |  |
| 7 | 7095 | this | 0.02893066 |  |
| 8 | 31672 | us | 0.02563477 |  |
| 9 | 22631 | time | 0.02563477 |  |
| 10 | 27564 | sometimes | 0.02258301 |  |
| 11 | 5892 | wolf | 0.01989746 |  |
| 12 | 6480 | he | 0.01989746 |  |
| 13 | 4726 | first | 0.01989746 |  |
| 14 | 37744 | last time | 0.01757812 |  |
| 15 | 16009 | opposition | 0.01757812 |  |
| 16 | 36935 | half body shot | 0.01550293 |  |
| 17 | 8477 | the | 0.01208496 |  |
| 18 | 5464 | seasons | 0.01208496 |  |
| 19 | 27385 | time | 0.01208496 |  |
| 20 | 11399 | and | 0.01068115 |  |
| 21 | 4765 | last | 0.01068115 |  |
| 22 | 27093 | piece | 0.01068115 |  |
| 23 | 5394 | basilica | 0.00939941 |  |
| 24 | 37544 | parliamentary group | 0.00939941 |  |
| 25 | 24797 | film | 0.00830078 |  |

---

## Exemplo 32

**Frase:**   
**Prompt:** `6480 13026 7212 8476 21988 8214`  
**Resposta correta:** ID `11693`  
**Resultado:** a resposta correta não apareceu no top-25.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 7074 | of | 0.97265625 |  |
| 2 | 36935 | half body shot | 0.02758789 |  |
| 3 | 7194 | for | 0.00016880 |  |
| 4 | 12313 | how | 0.00006247 |  |
| 5 | 5367 | next to | 0.00003433 |  |
| 6 | 37544 | parliamentary group | 0.00002360 |  |
| 7 | 7041 | to | 0.00002289 |  |
| 8 | 11709 | at the | 0.00000647 |  |
| 9 | 6152 | indigenous person of India | 0.00000566 |  |
| 10 | 6282 | indigenous person of Africa | 0.00000462 |  |
| 11 | 5958 | indigenous person of Asia | 0.00000374 |  |
| 12 | 16633 | care for | 0.00000322 |  |
| 13 | 11712 | of the | 0.00000320 |  |
| 14 | 37779 | somebody | 0.00000307 |  |
| 15 | 23751 | Noah's Ark | 0.00000149 |  |
| 16 | 6481 | he | 0.00000121 |  |
| 17 | 21854 | Statue of Liberty | 0.00000106 |  |
| 18 | 26769 | Democratic Republic of Congo | 0.00000083 |  |
| 19 | 6153 | indigenous person of the Americas | 0.00000068 |  |
| 20 | 26575 | Lutheran minister | 0.00000067 |  |
| 21 | 22629 | a lot of | 0.00000064 |  |
| 22 | 6642 | Poetry and Art's books | 0.00000063 |  |
| 23 | 7253 | solitary | 0.00000057 |  |
| 24 | 7803 | Who's calling? | 0.00000051 |  |
| 25 | 11691 | care for | 0.00000045 |  |

---

## Exemplo 33

**Frase:**   
**Prompt:** `7030 32669 37029 11709`  
**Resposta correta:** ID `4726`  
**Resultado:** a resposta correta não apareceu no top-25.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 24204 | building site | 0.80859375 |  |
| 2 | 24731 | how many? | 0.02600098 |  |
| 3 | 5939 | camping site | 0.02600098 |  |
| 4 | 5438 | in front | 0.01434326 |  |
| 5 | 36908 | gild | 0.01013184 |  |
| 6 | 2429 | infusion | 0.00842285 |  |
| 7 | 3418 | ? | 0.00765991 |  |
| 8 | 38570 | what is your name? | 0.00744629 |  |
| 9 | 19537 | building work | 0.00744629 |  |
| 10 | 2683 | countryside | 0.00698853 |  |
| 11 | 10369 | what program do you want to watch? | 0.00561523 |  |
| 12 | 32420 | camping site | 0.00479126 |  |
| 13 | 31334 | base-ten | 0.00463867 |  |
| 14 | 6283 | car park | 0.00424194 |  |
| 15 | 36259 | I don't see well | 0.00386047 |  |
| 16 | 6531 | have an idea | 0.00350952 |  |
| 17 | 36935 | half body shot | 0.00340271 |  |
| 18 | 37544 | parliamentary group | 0.00213623 |  |
| 19 | 16087 | job | 0.00213623 |  |
| 20 | 38216 | volume up button | 0.00205994 |  |
| 21 | 26585 | visit the tomb | 0.00199890 |  |
| 22 | 33074 | living room | 0.00187683 |  |
| 23 | 4714 | attack with a stick | 0.00165558 |  |
| 24 | 19570 | 1/2 | 0.00151062 |  |
| 25 | 28611 | spanish bar | 0.00146484 |  |

---

## Exemplo 34

**Frase:**   
**Prompt:** `8476 11246 8477 28677 7041`  
**Resposta correta:** ID `2704`  
**Resultado:** a resposta correta apareceu na posição **22**.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 8476 | the | 0.75781250 |  |
| 2 | 11712 | of the | 0.07275391 |  |
| 3 | 38413 | eat with the hand | 0.04248047 |  |
| 4 | 34939 | charge the battery | 0.03710938 |  |
| 5 | 7074 | of | 0.02404785 |  |
| 6 | 8477 | the | 0.01684570 |  |
| 7 | 38202 | clear all button | 0.00527954 |  |
| 8 | 7206 | burst the balloon | 0.00460815 |  |
| 9 | 3027 | f | 0.00193787 |  |
| 10 | 38333 | recharge the battery | 0.00175476 |  |
| 11 | 38033 | pull shirt collar | 0.00155640 |  |
| 12 | 2637 | open the box | 0.00150299 |  |
| 13 | 15497 | put the bra on | 0.00123596 |  |
| 14 | 5096 | put on cream | 0.00117493 |  |
| 15 | 7235 | roar | 0.00117493 |  |
| 16 | 37544 | parliamentary group | 0.00112915 |  |
| 17 | 11709 | at the | 0.00105286 |  |
| 18 | 34367 | get up from the floor | 0.00103760 |  |
| 19 | 12351 | realize | 0.00091553 |  |
| 20 | 32757 | put | 0.00075912 |  |
| 21 | 5569 | pull out | 0.00072479 |  |
| 22 | 2704 | city | 0.00071335 | ✅ |
| 23 | 25393 | gnaw | 0.00071335 |  |
| 24 | 36329 | order by phone | 0.00070190 |  |
| 25 | 3428 | ca-ca | 0.00069046 |  |

---

## Exemplo 35

**Frase:**   
**Prompt:** `6480 6190 12333 8476 4667`  
**Resposta correta:** ID `6903`  
**Resultado:** a resposta correta apareceu na posição **1**.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 6903 | year | 0.99609375 | ✅ |
| 2 | 37371 | day | 0.00204468 |  |
| 3 | 7022 | day | 0.00080109 |  |
| 4 | 7074 | of | 0.00040245 |  |
| 5 | 11399 | and | 0.00006771 |  |
| 6 | 11709 | at the | 0.00006199 |  |
| 7 | 15537 | University | 0.00005984 |  |
| 8 | 12313 | how | 0.00005817 |  |
| 9 | 6531 | have an idea | 0.00004816 |  |
| 10 | 11164 | lucky | 0.00003862 |  |
| 11 | 38570 | what is your name? | 0.00002503 |  |
| 12 | 7194 | for | 0.00001258 |  |
| 13 | 37724 | month | 0.00001043 |  |
| 14 | 7041 | to | 0.00000918 |  |
| 15 | 32862 | share an idea | 0.00000715 |  |
| 16 | 5464 | seasons | 0.00000653 |  |
| 17 | 38004 | essence | 0.00000557 |  |
| 18 | 11351 | that | 0.00000338 |  |
| 19 | 36935 | half body shot | 0.00000298 |  |
| 20 | 8476 | the | 0.00000289 |  |
| 21 | 7116 | people | 0.00000255 |  |
| 22 | 8153 | key | 0.00000247 |  |
| 23 | 15515 | high school | 0.00000247 |  |
| 24 | 5521 | a lot | 0.00000175 |  |
| 25 | 8261 | New Year's Eve | 0.00000150 |  |

---

## Exemplo 36

**Frase:**   
**Prompt:** `11377 11313 8476 37779 11313`  
**Resposta correta:** ID `37779`  
**Resultado:** a resposta correta não apareceu no top-25.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 8476 | the | 0.98437500 |  |
| 2 | 8477 | the | 0.01239014 |  |
| 3 | 11712 | of the | 0.00115204 |  |
| 4 | 12272 | his | 0.00016117 |  |
| 5 | 36935 | half body shot | 0.00004339 |  |
| 6 | 11709 | at the | 0.00003171 |  |
| 7 | 6481 | he | 0.00001872 |  |
| 8 | 6480 | he | 0.00001645 |  |
| 9 | 2637 | open the box | 0.00000966 |  |
| 10 | 12276 | his | 0.00000542 |  |
| 11 | 34939 | charge the battery | 0.00000533 |  |
| 12 | 5526 | no | 0.00000519 |  |
| 13 | 5800 | no | 0.00000444 |  |
| 14 | 10177 | sit on the toilet | 0.00000429 |  |
| 15 | 5397 | well | 0.00000423 |  |
| 16 | 9005 | tilt the bottle | 0.00000362 |  |
| 17 | 25708 | very | 0.00000335 |  |
| 18 | 37900 | slip the sand | 0.00000335 |  |
| 19 | 2675 | drop the glass | 0.00000300 |  |
| 20 | 7041 | to | 0.00000249 |  |
| 21 | 25036 | Christ the Redeemer | 0.00000191 |  |
| 22 | 12274 | theirs | 0.00000146 |  |
| 23 | 37589 | hake | 0.00000146 |  |
| 24 | 31790 | spin on the chair | 0.00000139 |  |
| 25 | 3033 | l | 0.00000131 |  |

---

## Exemplo 37

**Frase:**   
**Prompt:** `2627 6148 25708 32436`  
**Resposta correta:** ID `36434`  
**Resultado:** a resposta correta não apareceu no top-25.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 11377 | but | 0.72265625 |  |
| 2 | 11399 | and | 0.15625000 |  |
| 3 | 7041 | to | 0.09472656 |  |
| 4 | 7074 | of | 0.01361084 |  |
| 5 | 36480 | be | 0.00543213 |  |
| 6 | 35949 | be able to | 0.00215149 |  |
| 7 | 11698 | produce | 0.00209045 |  |
| 8 | 6906 | that one | 0.00122833 |  |
| 9 | 11709 | at the | 0.00093460 |  |
| 10 | 6531 | have an idea | 0.00026131 |  |
| 11 | 36329 | order by phone | 0.00024986 |  |
| 12 | 3417 | ! | 0.00020218 |  |
| 13 | 36377 | be accountable to | 0.00015068 |  |
| 14 | 11351 | that | 0.00012112 |  |
| 15 | 7194 | for | 0.00008965 |  |
| 16 | 38570 | what is your name? | 0.00008631 |  |
| 17 | 8555 | turn on the monitor | 0.00008154 |  |
| 18 | 11712 | of the | 0.00007057 |  |
| 19 | 32816 | wish for | 0.00006294 |  |
| 20 | 21793 | say goodbye | 0.00006104 |  |
| 21 | 39069 | prick | 0.00005698 |  |
| 22 | 2627 | one | 0.00005627 |  |
| 23 | 2872 | arrange | 0.00005579 |  |
| 24 | 5581 | be | 0.00005269 |  |
| 25 | 15497 | put the bra on | 0.00004172 |  |

---

## Exemplo 38

**Frase:**   
**Prompt:** `12272 38570`  
**Resposta correta:** ID `36935`  
**Resultado:** a resposta correta não apareceu no top-25.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 7074 | of | 0.21093750 |  |
| 2 | 6190 | prepare | 0.12792969 |  |
| 3 | 36480 | be | 0.09960938 |  |
| 4 | 32669 | come | 0.06835938 |  |
| 5 | 17040 | give | 0.03662109 |  |
| 6 | 37369 | remember | 0.03662109 |  |
| 7 | 11709 | at the | 0.02856445 |  |
| 8 | 32761 | have | 0.02526855 |  |
| 9 | 6517 | speak | 0.02526855 |  |
| 10 | 37029 | frequently | 0.02221680 |  |
| 11 | 9693 | say | 0.01531982 |  |
| 12 | 32324 | associate | 0.01190186 |  |
| 13 | 32806 | ancient | 0.01190186 |  |
| 14 | 7041 | to | 0.01190186 |  |
| 15 | 9848 | present | 0.01190186 |  |
| 16 | 32705 | english | 0.01049805 |  |
| 17 | 5451 | above | 0.00817871 |  |
| 18 | 8476 | the | 0.00817871 |  |
| 19 | 16823 | show | 0.00817871 |  |
| 20 | 29129 | reject | 0.00817871 |  |
| 21 | 30510 | choose | 0.00723267 |  |
| 22 | 9901 | sign on | 0.00723267 |  |
| 23 | 37802 | famous | 0.00723267 |  |
| 24 | 5431 | begin | 0.00637817 |  |
| 25 | 2380 | write | 0.00637817 |  |

---

## Exemplo 39

**Frase:**   
**Prompt:** `29550 8076 6624 7041 22631`  
**Resposta correta:** ID `26176`  
**Resultado:** a resposta correta não apareceu no top-25.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 7212 | by | 0.60546875 |  |
| 2 | 8476 | the | 0.18652344 |  |
| 3 | 7041 | to | 0.05273438 |  |
| 4 | 5367 | next to | 0.05053711 |  |
| 5 | 11709 | at the | 0.04882812 |  |
| 6 | 7081 | during | 0.01324463 |  |
| 7 | 2628 | 2 | 0.01116943 |  |
| 8 | 11399 | and | 0.00497437 |  |
| 9 | 7810 | according to | 0.00393677 |  |
| 10 | 7064 | with | 0.00354004 |  |
| 11 | 19548 | 1-2-3 | 0.00297546 |  |
| 12 | 12274 | theirs | 0.00183105 |  |
| 13 | 7765 | between | 0.00159454 |  |
| 14 | 7034 | in | 0.00151825 |  |
| 15 | 24737 | during | 0.00142670 |  |
| 16 | 7074 | of | 0.00129700 |  |
| 17 | 19570 | 1/2 | 0.00111389 |  |
| 18 | 8477 | the | 0.00109100 |  |
| 19 | 31670 | It | 0.00107574 |  |
| 20 | 36935 | half body shot | 0.00075150 |  |
| 21 | 5451 | above | 0.00071716 |  |
| 22 | 2630 | 4 | 0.00057602 |  |
| 23 | 37544 | parliamentary group | 0.00045013 |  |
| 24 | 11712 | of the | 0.00045013 |  |
| 25 | 10271 | Eiffel Tower | 0.00033951 |  |

---

## Exemplo 40

**Frase:**   
**Prompt:** `2627 37371 36935 36449`  
**Resposta correta:** ID `21565`  
**Resultado:** a resposta correta apareceu na posição **1**.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 21565 | hurt the hand | 0.19921875 | ✅ |
| 2 | 2376 | get sick | 0.15136719 |  |
| 3 | 13024 | hand in hand | 0.13085938 |  |
| 4 | 35573 | tiredness | 0.09375000 |  |
| 5 | 30636 | look down on | 0.09082031 |  |
| 6 | 31004 | look down on | 0.06127930 |  |
| 7 | 33056 | rudder | 0.04638672 |  |
| 8 | 25752 | you got it! | 0.02832031 |  |
| 9 | 27393 | interrupt | 0.02185059 |  |
| 10 | 8558 | illness | 0.02001953 |  |
| 11 | 36935 | half body shot | 0.01507568 |  |
| 12 | 11399 | and | 0.01403809 |  |
| 13 | 38413 | eat with the hand | 0.01403809 |  |
| 14 | 4714 | attack with a stick | 0.00781250 |  |
| 15 | 12315 | comply | 0.00561523 |  |
| 16 | 7040 | get sick | 0.00537109 |  |
| 17 | 34367 | get up from the floor | 0.00485229 |  |
| 18 | 6531 | have an idea | 0.00317383 |  |
| 19 | 15497 | put the bra on | 0.00309753 |  |
| 20 | 37420 | get angry | 0.00300598 |  |
| 21 | 6642 | Poetry and Art's books | 0.00288391 |  |
| 22 | 36329 | order by phone | 0.00288391 |  |
| 23 | 24929 | drunkenness | 0.00285339 |  |
| 24 | 9072 | pick up a package | 0.00267029 |  |
| 25 | 31873 | hold a wake over | 0.00247192 |  |

---

## Exemplo 41

**Frase:**   
**Prompt:** `21565 6480 26954 7074 8474`  
**Resposta correta:** ID `8666`  
**Resultado:** a resposta correta apareceu na posição **1**.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 8666 | leg | 0.77734375 | ✅ |
| 2 | 27953 | leg | 0.19628906 |  |
| 3 | 37166 | leg | 0.00976562 |  |
| 4 | 16833 | leg | 0.00976562 |  |
| 5 | 28769 | leg pain | 0.00421143 |  |
| 6 | 28770 | leg pain | 0.00180817 |  |
| 7 | 5484 | injury | 0.00096893 |  |
| 8 | 2904 | wrist | 0.00013065 |  |
| 9 | 3000 | hip | 0.00012684 |  |
| 10 | 11399 | and | 0.00008440 |  |
| 11 | 3241 | ball | 0.00007820 |  |
| 12 | 8729 | wooden leg | 0.00007010 |  |
| 13 | 3405 | ankle | 0.00002348 |  |
| 14 | 7074 | of | 0.00002074 |  |
| 15 | 36777 | ny | 0.00002074 |  |
| 16 | 35809 | foot pain | 0.00000936 |  |
| 17 | 2972 | bone | 0.00000876 |  |
| 18 | 8558 | illness | 0.00000432 |  |
| 19 | 2269 | ball | 0.00000408 |  |
| 20 | 39084 | leg up | 0.00000334 |  |
| 21 | 25327 | foot | 0.00000305 |  |
| 22 | 34090 | bat and ball | 0.00000261 |  |
| 23 | 6528 | bone | 0.00000240 |  |
| 24 | 31773 | urinary tract infection | 0.00000180 |  |
| 25 | 11008 | wool ball | 0.00000177 |  |

---

## Exemplo 42

**Frase:**   
**Prompt:** `9839 6632 5397 37721 7041 6564 8476`  
**Resposta correta:** ID `37704`  
**Resultado:** a resposta correta apareceu na posição **9**.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 3118 | church | 0.53125000 |  |
| 2 | 38464 | cabaret | 0.24316406 |  |
| 3 | 6530 | church | 0.05004883 |  |
| 4 | 2933 | Moon | 0.02648926 |  |
| 5 | 2704 | city | 0.02160645 |  |
| 6 | 6609 | handshake | 0.01623535 |  |
| 7 | 11690 | creativity | 0.01391602 |  |
| 8 | 10238 | ox | 0.01068115 |  |
| 9 | 37704 | owner | 0.00830078 | ✅ |
| 10 | 3244 | door | 0.00561523 |  |
| 11 | 16591 | loot | 0.00445557 |  |
| 12 | 8296 | christmas ball | 0.00415039 |  |
| 13 | 29947 | game of yo-yo | 0.00379944 |  |
| 14 | 36153 | melon | 0.00291443 |  |
| 15 | 7074 | of | 0.00245667 |  |
| 16 | 33036 | maid | 0.00205994 |  |
| 17 | 3322 | jar | 0.00178528 |  |
| 18 | 2925 | sea | 0.00164795 |  |
| 19 | 9810 | games | 0.00146484 |  |
| 20 | 2264 | aeroplane | 0.00144196 |  |
| 21 | 11008 | wool ball | 0.00140381 |  |
| 22 | 6065 | bin bag | 0.00122070 |  |
| 23 | 38506 | pet bowl | 0.00117493 |  |
| 24 | 5561 | clock | 0.00110626 |  |
| 25 | 38249 | back button | 0.00091171 |  |

---

## Exemplo 43

**Frase:**   
**Prompt:** `15433 7074 2627 23731`  
**Resposta correta:** ID `33994`  
**Resultado:** a resposta correta não apareceu no top-25.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 11399 | and | 0.95312500 |  |
| 2 | 10217 | lazy | 0.02380371 |  |
| 3 | 31352 | whistle | 0.00515747 |  |
| 4 | 7249 | whistle | 0.00332642 |  |
| 5 | 5523 | whipped cream | 0.00236511 |  |
| 6 | 35363 | whistle | 0.00122833 |  |
| 7 | 38560 | it's sunny! | 0.00069809 |  |
| 8 | 34637 | undressed | 0.00061798 |  |
| 9 | 7074 | of | 0.00046539 |  |
| 10 | 25249 | mucus | 0.00045013 |  |
| 11 | 2885 | misty | 0.00045013 |  |
| 12 | 30485 | anxious | 0.00039864 |  |
| 13 | 11332 | hairy | 0.00033951 |  |
| 14 | 7773 | spear | 0.00033951 |  |
| 15 | 14254 | saliva | 0.00032043 |  |
| 16 | 37699 | blow up | 0.00032043 |  |
| 17 | 25353 | alight | 0.00030899 |  |
| 18 | 35209 | ice-cream | 0.00029945 |  |
| 19 | 34603 | pour whipped cream | 0.00029182 |  |
| 20 | 7064 | with | 0.00026512 |  |
| 21 | 38811 | furious | 0.00024605 |  |
| 22 | 8103 | turn on the light | 0.00019073 |  |
| 23 | 24711 | be nice | 0.00017071 |  |
| 24 | 15411 | praxia | 0.00012493 |  |
| 25 | 15397 | praxia | 0.00012493 |  |

---

## Exemplo 44

**Frase:**   
**Prompt:** `9839 36935 32761 7194 7041 8476 33068`  
**Resposta correta:** ID `36935`  
**Resultado:** a resposta correta apareceu na posição **4**.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 37779 | somebody | 0.42773438 |  |
| 2 | 26575 | Lutheran minister | 0.12255859 |  |
| 3 | 22757 | administrator | 0.09667969 |  |
| 4 | 36935 | half body shot | 0.07177734 | ✅ |
| 5 | 37577 | social worker | 0.03369141 |  |
| 6 | 35884 | secret ballot | 0.03063965 |  |
| 7 | 3036 | n | 0.03039551 |  |
| 8 | 4553 | put to bed | 0.01684570 |  |
| 9 | 6642 | Poetry and Art's books | 0.01556396 |  |
| 10 | 32816 | wish for | 0.01312256 |  |
| 11 | 5526 | no | 0.00781250 |  |
| 12 | 10196 | littler | 0.00708008 |  |
| 13 | 23745 | apostle | 0.00686646 |  |
| 14 | 24829 | riddle | 0.00521851 |  |
| 15 | 5800 | no | 0.00479126 |  |
| 16 | 22629 | a lot of | 0.00421143 |  |
| 17 | 11709 | at the | 0.00415039 |  |
| 18 | 34637 | undressed | 0.00367737 |  |
| 19 | 10366 | what is that? | 0.00346375 |  |
| 20 | 2746 | listen to music | 0.00346375 |  |
| 21 | 9072 | pick up a package | 0.00338745 |  |
| 22 | 7264 | put the top on | 0.00337219 |  |
| 23 | 6010 | Arabian | 0.00265503 |  |
| 24 | 7223 | what's the weather like? | 0.00209045 |  |
| 25 | 7074 | of | 0.00185394 |  |

---

## Exemplo 45

**Frase:**   
**Prompt:** `12276 35062 36480 5374 38444`  
**Resposta correta:** ID `30642`  
**Resultado:** a resposta correta não apareceu no top-25.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 7074 | of | 0.90234375 |  |
| 2 | 7041 | to | 0.09521484 |  |
| 3 | 12333 | organization | 0.00025177 |  |
| 4 | 30042 | secular | 0.00025177 |  |
| 5 | 36942 | nobles | 0.00016212 |  |
| 6 | 8040 | Belgium | 0.00004101 |  |
| 7 | 23398 | interculturality | 0.00002646 |  |
| 8 | 11709 | at the | 0.00002337 |  |
| 9 | 7813 | without | 0.00001705 |  |
| 10 | 7194 | for | 0.00000834 |  |
| 11 | 35949 | be able to | 0.00000671 |  |
| 12 | 35457 | Jewish cementary | 0.00000572 |  |
| 13 | 37803 | rich | 0.00000507 |  |
| 14 | 12274 | theirs | 0.00000474 |  |
| 15 | 11712 | of the | 0.00000459 |  |
| 16 | 11399 | and | 0.00000459 |  |
| 17 | 37236 | sincerity | 0.00000432 |  |
| 18 | 2746 | listen to music | 0.00000337 |  |
| 19 | 12278 | theirs | 0.00000279 |  |
| 20 | 34781 | make fun of | 0.00000262 |  |
| 21 | 38033 | pull shirt collar | 0.00000262 |  |
| 22 | 16731 | blacksmith's | 0.00000169 |  |
| 23 | 34292 | mobilize | 0.00000164 |  |
| 24 | 8476 | the | 0.00000159 |  |
| 25 | 22629 | a lot of | 0.00000150 |  |

---

## Exemplo 46

**Frase:**   
**Prompt:** `6480 36480 7074 8476 37822 7074 7041`  
**Resposta correta:** ID `2704`  
**Resultado:** a resposta correta apareceu na posição **1**.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 2704 | city | 0.62500000 | ✅ |
| 2 | 23769 | Atomium | 0.10205078 |  |
| 3 | 21955 | karate | 0.05126953 |  |
| 4 | 37544 | parliamentary group | 0.04809570 |  |
| 5 | 21854 | Statue of Liberty | 0.02661133 |  |
| 6 | 21445 | Big Ben | 0.02575684 |  |
| 7 | 8177 | Navarre | 0.01220703 |  |
| 8 | 27810 | granary | 0.01184082 |  |
| 9 | 10271 | Eiffel Tower | 0.00866699 |  |
| 10 | 12333 | organization | 0.00811768 |  |
| 11 | 39105 | sweet granadilla | 0.00787354 |  |
| 12 | 8112 | United States of America | 0.00738525 |  |
| 13 | 37145 | calyx | 0.00695801 |  |
| 14 | 3372 | pistachios | 0.00631714 |  |
| 15 | 7093 | sword | 0.00299072 |  |
| 16 | 8474 | a | 0.00254822 |  |
| 17 | 36656 | military man | 0.00247192 |  |
| 18 | 6229 | treasure | 0.00247192 |  |
| 19 | 9185 | Point of Sale Terminal | 0.00239563 |  |
| 20 | 5064 | Olympic games | 0.00212097 |  |
| 21 | 3055 | flippers | 0.00186920 |  |
| 22 | 14682 | Saragossa | 0.00154877 |  |
| 23 | 2913 | tape measure | 0.00120544 |  |
| 24 | 35177 | human resources | 0.00120544 |  |
| 25 | 2831 | pistol | 0.00116730 |  |

---

## Exemplo 47

**Frase:**   
**Prompt:** `32747 7028`  
**Resposta correta:** ID `35235`  
**Resultado:** a resposta correta não apareceu no top-25.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 37802 | famous | 0.51171875 |  |
| 2 | 5508 | more | 0.07861328 |  |
| 3 | 32761 | have | 0.04760742 |  |
| 4 | 8476 | the | 0.04199219 |  |
| 5 | 8474 | a | 0.03710938 |  |
| 6 | 11399 | and | 0.02551270 |  |
| 7 | 36480 | be | 0.02551270 |  |
| 8 | 2627 | one | 0.01550293 |  |
| 9 | 15523 | obligation | 0.01367188 |  |
| 10 | 35949 | be able to | 0.01367188 |  |
| 11 | 8456 | know | 0.00939941 |  |
| 12 | 15485 | use | 0.00830078 |  |
| 13 | 36849 | spray | 0.00830078 |  |
| 14 | 6481 | he | 0.00830078 |  |
| 15 | 11180 | squash | 0.00646973 |  |
| 16 | 6624 | work | 0.00570679 |  |
| 17 | 6517 | speak | 0.00570679 |  |
| 18 | 6190 | prepare | 0.00503540 |  |
| 19 | 6564 | see | 0.00503540 |  |
| 20 | 7041 | to | 0.00503540 |  |
| 21 | 11709 | at the | 0.00442505 |  |
| 22 | 10203 | famous | 0.00390625 |  |
| 23 | 10202 | famous | 0.00390625 |  |
| 24 | 5596 | all | 0.00390625 |  |
| 25 | 8477 | the | 0.00390625 |  |

---

## Exemplo 48

**Frase:**   
**Prompt:** `5382 9839 8476 5563 16823 27705 11399`  
**Resposta correta:** ID `26985`  
**Resultado:** a resposta correta não apareceu no top-25.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 26909 | ingenuous | 0.16601562 |  |
| 2 | 6642 | Poetry and Art's books | 0.09960938 |  |
| 3 | 11554 | homosexual | 0.08300781 |  |
| 4 | 2746 | listen to music | 0.08007812 |  |
| 5 | 35949 | be able to | 0.07080078 |  |
| 6 | 38811 | furious | 0.04248047 |  |
| 7 | 37036 | take to school | 0.03442383 |  |
| 8 | 27705 | curious | 0.02343750 |  |
| 9 | 30630 | jealous | 0.01586914 |  |
| 10 | 3027 | f | 0.01104736 |  |
| 11 | 32753 | I want more | 0.01074219 |  |
| 12 | 36273 | therapeutic pedagogy teacher | 0.00738525 |  |
| 13 | 17253 | hyena | 0.00738525 |  |
| 14 | 32824 | special education school | 0.00714111 |  |
| 15 | 34637 | undressed | 0.00692749 |  |
| 16 | 15515 | high school | 0.00692749 |  |
| 17 | 9072 | pick up a package | 0.00662231 |  |
| 18 | 3428 | ca-ca | 0.00619507 |  |
| 19 | 6160 | ogre | 0.00592041 |  |
| 20 | 34790 | lend a book | 0.00592041 |  |
| 21 | 37236 | sincerity | 0.00582886 |  |
| 22 | 6599 | defend | 0.00582886 |  |
| 23 | 32308 | I am not able to read | 0.00564575 |  |
| 24 | 25433 | sauna | 0.00555420 |  |
| 25 | 2637 | open the box | 0.00555420 |  |

---

## Exemplo 49

**Frase:**   
**Prompt:** `8476 8456 12313 7253 2944 7074 8476`  
**Resposta correta:** ID `8532`  
**Resultado:** a resposta correta não apareceu no top-25.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 26666 | Andorra | 0.65234375 |  |
| 2 | 25776 | isolate | 0.25976562 |  |
| 3 | 28979 | Andorra | 0.02832031 |  |
| 4 | 9082 | UNESCO | 0.01550293 |  |
| 5 | 9815 | classroom | 0.01159668 |  |
| 6 | 32806 | ancient | 0.00665283 |  |
| 7 | 5913 | classroom | 0.00318909 |  |
| 8 | 36771 | ű | 0.00207520 |  |
| 9 | 8098 | education | 0.00154114 |  |
| 10 | 32824 | special education school | 0.00107574 |  |
| 11 | 11245 | artistic education | 0.00101471 |  |
| 12 | 26872 | Georgia | 0.00054932 |  |
| 13 | 21873 | Finland | 0.00053406 |  |
| 14 | 38021 | education | 0.00051498 |  |
| 15 | 9101 | crystalline lens | 0.00048637 |  |
| 16 | 30014 | The Earth | 0.00047874 |  |
| 17 | 11693 | culture | 0.00046349 |  |
| 18 | 14834 | Lerida | 0.00043488 |  |
| 19 | 23398 | interculturality | 0.00037766 |  |
| 20 | 17332 | viola | 0.00034332 |  |
| 21 | 33082 | theater classroom | 0.00033379 |  |
| 22 | 6642 | Poetry and Art's books | 0.00029373 |  |
| 23 | 36897 | Burgos' Cathedral | 0.00029373 |  |
| 24 | 31985 | vocational education | 0.00028610 |  |
| 25 | 23751 | Noah's Ark | 0.00025940 |  |

---

## Exemplo 50

**Frase:**   
**Prompt:** `8476 11578 7074 7092 36964 11578 7074`  
**Resposta correta:** ID `22161`  
**Resultado:** a resposta correta não apareceu no top-25.

| Posição | ID | Texto | Probabilidade | Correto |
|---:|---:|---|---:|:---:|
| 1 | 2704 | city | 0.53906250 |  |
| 2 | 37544 | parliamentary group | 0.40625000 |  |
| 3 | 37779 | somebody | 0.02294922 |  |
| 4 | 8112 | United States of America | 0.01306152 |  |
| 5 | 8476 | the | 0.00588989 |  |
| 6 | 27798 | globe | 0.00205994 |  |
| 7 | 6282 | indigenous person of Africa | 0.00202942 |  |
| 8 | 37406 | write the title | 0.00101471 |  |
| 9 | 30014 | The Earth | 0.00096893 |  |
| 10 | 6152 | indigenous person of India | 0.00093842 |  |
| 11 | 5958 | indigenous person of Asia | 0.00052261 |  |
| 12 | 5367 | next to | 0.00041389 |  |
| 13 | 7116 | people | 0.00038910 |  |
| 14 | 6153 | indigenous person of the Americas | 0.00035286 |  |
| 15 | 8144 | Japan | 0.00029373 |  |
| 16 | 6137 | indigenous person of the Americas | 0.00027847 |  |
| 17 | 29919 | sports club | 0.00024223 |  |
| 18 | 29598 | Chile | 0.00021839 |  |
| 19 | 6642 | Poetry and Art's books | 0.00015926 |  |
| 20 | 26845 | prize | 0.00015831 |  |
| 21 | 5087 | ski run | 0.00015640 |  |
| 22 | 35129 | chess board | 0.00015450 |  |
| 23 | 2627 | one | 0.00014973 |  |
| 24 | 8206 | United Kingdom | 0.00014496 |  |
| 25 | 11205 | running race | 0.00014210 |  |

---

# Resumo

- Exemplos válidos analisados: 50
- Respostas encontradas no top-25: 22
- `accuracy@25`: 0.440000
- Posição média da resposta correta: 6.64
- Melhor posição encontrada: 1
- Pior posição encontrada: 22

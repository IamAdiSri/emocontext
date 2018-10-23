# Format for Pickled Data

## hedges.pkl

Contains the Hyland's list of hedges and their inflections.

```
((list)) {
    HEDGE ((str)), 
    .
    .
}
```

## factives.pkl

Contains Hooper's list of factive verbs and their inflections.

```
((list)) {
    FACTIVE ((str)), 
    .
    .
}
```

## assertives.pkl

Contains Hooper's list of assertive verbs and their inflections.

```
((list)) {
    ASSERTIVE ((str)), 
    .
    .
}
```

## implicatives.pkl

Contains Karttunen's list of implicative verbs and their inflections.

```
((list)) {
    IMPLICATIVE ((str)), 
    .
    .
}
```

## reports.pkl

Contains list of report verbs and their inflections.

```
((list)) {
    REPORT ((str)), 
    .
    .
}
```

## entailments.pkl

Contains Berant et al.'s list of entailment causing verbs.

```
((list)) {
    ENTAILMENT ((str)), 
    .
    .
}
```

## subjectives.pkl

Contains Riloff and Wiebe's list of subjective verbs.

```
((dict)) {
    SUBJECTIVE ((str))  : ((dict)) {
                              'type'    : STRENGTH  ((str)), 
                              'len'     : LENGTH    ((str)), 
                              'pos'     : POS_TAG   ((str)), 
                              'stem'    : Y/N       ((str)),
                              'pol'     : POLARITY  ((str)), 
                          }, 
    .
    .
}
```
**Example**

`{ 'abandon' : {'type': 'weaksubj', 'len': '1', 'pos': 'verb', 'stem': 'y', 'pol': 'negative'} }`

## polarity.pkl

Contains Liu et al.'s lists of positive and negative words.

```
((dict)) {
    WORD ((str))        : POLARITY ((str))
    .
    .
}
```

Polarity can be either `'positive'` or `'negative'`.
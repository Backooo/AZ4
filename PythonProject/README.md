# PythonProject
Test Repository: PythonProject

Notizen für mich:

MCTS - Monte Carlo Tree Search Algo...

ToDo:
-   Ideen suchen für die verbesserung der Leistung:
    - weit von einander entferte columns sinnloser als zusammenhängende? -> gewisse züge priorisieren über andere? (column 1 -> 6 -> 3 -> 0 -> 4 ->...., das würde dem anderem eh zu viel zeit geben sich 4 nebeneinander zu bauen und damit brauch man solche fälle nicht immer berechnen (wie viel schlechter wird der Agent dann, im austausch zu wie viel performence?))
    -mittig im spielbrett mehr möglichkeiten -> diese züge bevorzugen??
    - in der nähe des gegners simulieren und dann generell (zwei verschiedene algos?) -> irgendwie schaffen das der immer min. 4 schritte weiter weiß was passieren wird wenn der gegner einen move macht
    -   wenn len(self.children) = 7 dann kinder anders auswählen (aktuell random.choice(self.children).create_children())
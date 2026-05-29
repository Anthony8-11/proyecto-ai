% rules.pl
% Motor de restricciones simbólico — Sistema Autónomo de RL
% Sintaxis estándar ISO Prolog / SWI-Prolog 10
%
% Interfaz pública:
%   no_accion(+Accion, +Pos, +Rec, +Cond)   — verdadero si la acción está prohibida
%   accion_permitida(+Accion, +Pos, +Rec, +Cond) — verdadero si la acción está permitida
%
% Átomos de acción: mover | asignar | esperar | reaccionar
% Condición del entorno:  0 = normal · 1 = alerta · 2 = crítico

:- module(rules, [no_accion/4, accion_permitida/4]).

% ---------------------------------------------------------------------------
% Constantes del dominio
% ---------------------------------------------------------------------------
min_recursos(2).
critico(2).
alerta(1).

% ---------------------------------------------------------------------------
% Reglas de prohibición (10 reglas de predicados)
% ---------------------------------------------------------------------------

% R1: No moverse si los recursos son menores al mínimo permitido
no_accion(mover, _Pos, Rec, _Cond) :-
    min_recursos(Min),
    Rec < Min.

% R2: No moverse en condición crítica (riesgo de fallo total)
no_accion(mover, _Pos, _Rec, Cond) :-
    critico(Crit),
    Cond =:= Crit.

% R3: No asignar recursos cuando ya se está en el máximo
no_accion(asignar, _Pos, Rec, _Cond) :-
    Rec >= 10.

% R4: Bloqueo total en estado crítico sin recursos disponibles
no_accion(asignar, _Pos, Rec, Cond) :-
    critico(Crit),
    Cond =:= Crit,
    Rec < 1.

% R5: Esperar en condición crítica es una acción inválida
no_accion(esperar, _Pos, _Rec, Cond) :-
    critico(Crit),
    Cond =:= Crit.

% R6: No esperar con recursos casi agotados en alerta o crítico
no_accion(esperar, _Pos, Rec, Cond) :-
    alerta(Alert),
    Rec =< 1,
    Cond >= Alert.

% R7: Reaccionar sin amenaza activa es inútil (condición normal = 0)
no_accion(reaccionar, _Pos, _Rec, 0).

% R8: No se puede reaccionar sin recursos disponibles
no_accion(reaccionar, _Pos, Rec, _Cond) :-
    Rec < 1.

% R9: En posición 0, solo reaccionar si la condición es de alerta o superior
no_accion(reaccionar, 0, _Rec, Cond) :-
    alerta(Alert),
    Cond < Alert.

% R10: Con recursos abundantes y entorno en calma, reaccionar es innecesario
no_accion(reaccionar, _Pos, Rec, 0) :-
    Rec > 8.

% ---------------------------------------------------------------------------
% Regla principal de permisión
% ---------------------------------------------------------------------------

% Una acción es permitida si y solo si no existe ninguna regla que la prohíba.
% Usa negación por fallo (\+) — estándar en Prolog ISO.
accion_permitida(Accion, Pos, Rec, Cond) :-
    \+ no_accion(Accion, Pos, Rec, Cond).

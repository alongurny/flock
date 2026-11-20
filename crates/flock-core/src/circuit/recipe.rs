//! A circuit as data a verifier can rebuild: the INPUTS of its validating
//! constructors, never its derived tables.
//!
//! A verification key that ships a circuit must not let the shipper choose
//! anything the constructors would have computed (the cell space, sigma,
//! the canonical wiring, the registry's slot layout). So a
//! [`CircuitRecipe`] holds only what [`Registry::new`],
//! [`Circuit::new_with_fixed_public`] and the `with_*` steps take, in the
//! builder's order (pruning, public columns, slot packing, then the bus),
//! and [`CircuitRecipe::rebuild`] runs them again: every check they make
//! (live cells, disjoint classes, acyclic dataflow, sigma a permutation,
//! the bus fits) is made again, and the digest is recomputed from the
//! result. A recipe is honest exactly when its rebuilt digest is the one
//! the verifier pins.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::{Cell, Circuit, CircuitError, bus::Bus};
use crate::{
    element_r1cs::{ElementTableType, SparseF128Matrix},
    field::F128,
    r1cs::SparseBinaryMatrix,
    schedule::{IoDirection, IoWord, Registry, TableClass, TableType},
};

/// A sparse binary matrix: its shape and each row's column indices.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatrixRecipe {
    pub num_rows: u32,
    pub num_cols: u32,
    pub rows: Vec<Vec<u32>>,
}

impl MatrixRecipe {
    fn of(m: &SparseBinaryMatrix) -> Self {
        Self {
            num_rows: u32::try_from(m.num_rows).expect("rows fit u32"),
            num_cols: u32::try_from(m.num_cols).expect("columns fit u32"),
            rows: m
                .rows
                .iter()
                .map(|r| r.iter().map(|&c| c as u32).collect())
                .collect(),
        }
    }

    fn rebuild(&self) -> SparseBinaryMatrix {
        SparseBinaryMatrix::new(
            self.num_rows as usize,
            self.num_cols as usize,
            self.rows
                .iter()
                .map(|r| r.iter().map(|&c| c as usize).collect())
                .collect(),
        )
    }
}

/// A sparse matrix over F128: its shape and each row's `(column, entry)`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct F128MatrixRecipe {
    pub num_rows: u32,
    pub num_cols: u32,
    pub rows: Vec<Vec<(u32, F128)>>,
}

impl F128MatrixRecipe {
    fn of(m: &SparseF128Matrix) -> Self {
        Self {
            num_rows: u32::try_from(m.num_rows).expect("rows fit u32"),
            num_cols: u32::try_from(m.num_cols).expect("columns fit u32"),
            rows: m
                .rows
                .iter()
                .map(|r| r.iter().map(|&(c, w)| (c as u32, w)).collect())
                .collect(),
        }
    }

    fn rebuild(&self) -> SparseF128Matrix {
        SparseF128Matrix {
            num_rows: self.num_rows as usize,
            num_cols: self.num_cols as usize,
            rows: self
                .rows
                .iter()
                .map(|r| r.iter().map(|&(c, w)| (c as usize, w)).collect())
                .collect(),
        }
    }
}

/// An element table type: the arguments of [`ElementTableType::new`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ElementRecipe {
    pub kappa: u32,
    pub k: u32,
    pub a_0: F128MatrixRecipe,
    pub b_0: F128MatrixRecipe,
    pub a_const: Vec<F128>,
    pub b_const: Vec<F128>,
}

/// One registry type. A Boolean type carries its three matrices; an
/// element type only its element table (its Boolean matrices are the
/// constructor's empty stubs).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TypeClassRecipe {
    Boolean {
        k_log: u32,
        useful_bits: u64,
        a_0: MatrixRecipe,
        b_0: MatrixRecipe,
        c_0: MatrixRecipe,
        const_pin: Option<u32>,
    },
    Element(ElementRecipe),
    /// A type the verifier's code builds itself (a large constant table,
    /// such as the BLAKE3 block), by name and row bits: see
    /// [`RegistryRecipe::rebuild_with`].
    Named { name: String, nu: u32 },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypeRecipe {
    pub class: TypeClassRecipe,
    /// `(word column, is output)` per schema word.
    pub io_schema: Vec<(u32, bool)>,
    pub zeroed_word_cols: Vec<u32>,
}

/// Why a recipe did not rebuild.
#[derive(Debug)]
pub enum RecipeError {
    /// An element table refused its constructor.
    Element(String),
    /// The circuit refused its constructor or its bus.
    Circuit(CircuitError),
    /// The rebuilt registry is not in the recipe's order (the recipe was
    /// not taken from a registry).
    RegistryOrder,
    /// A field does not fit the platform's `usize` or is out of range.
    Range,
}

/// A registry: its types in slot order and its row bits.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegistryRecipe {
    pub nu: u32,
    pub types: Vec<TypeRecipe>,
}

/// Names the types a verifier builds from code: `Some((name, nu))` for a
/// type to ship by name.
pub type Namer<'a> = &'a dyn Fn(&TableType) -> Option<(String, usize)>;
/// Builds a named type: the type `(name, nu)` names, if the verifier knows it.
pub type Resolver<'a> = &'a dyn Fn(&str, usize) -> Option<TableType>;

impl RegistryRecipe {
    pub fn of(r: &Registry) -> Self {
        Self::of_with(r, &|_| None)
    }

    pub fn of_with(r: &Registry, name: Namer<'_>) -> Self {
        let types = r
            .types()
            .iter()
            .map(|ty| {
                let class = if let Some((name, nu)) = name(ty) {
                    TypeClassRecipe::Named {
                        name,
                        nu: nu as u32,
                    }
                } else {
                    match ty.element_type() {
                    Some(e) => TypeClassRecipe::Element(ElementRecipe {
                        kappa: e.kappa() as u32,
                        k: e.k() as u32,
                        a_0: F128MatrixRecipe::of(e.a_0()),
                        b_0: F128MatrixRecipe::of(e.b_0()),
                        a_const: e.a_const().to_vec(),
                        b_const: e.b_const().to_vec(),
                    }),
                    None => TypeClassRecipe::Boolean {
                        k_log: ty.k_log as u32,
                        useful_bits: ty.useful_bits as u64,
                        a_0: MatrixRecipe::of(&ty.a_0),
                        b_0: MatrixRecipe::of(&ty.b_0),
                        c_0: MatrixRecipe::of(&ty.c_0),
                        const_pin: ty.const_pin.map(|c| c as u32),
                    },
                    }
                };
                TypeRecipe {
                    class,
                    io_schema: ty
                        .io_schema
                        .iter()
                        .map(|w| (w.word_col as u32, w.dir == IoDirection::Out))
                        .collect(),
                    zeroed_word_cols: ty.zeroed_word_cols.iter().map(|&w| w as u32).collect(),
                }
            })
            .collect();
        Self {
            nu: r.nu() as u32,
            types,
        }
    }

    pub fn rebuild(&self) -> Result<Registry, RecipeError> {
        self.rebuild_with(&|_, _| None)
    }

    pub fn rebuild_with(&self, resolve: Resolver<'_>) -> Result<Registry, RecipeError> {
        let mut types = Vec::with_capacity(self.types.len());
        for t in &self.types {
            let mut ty = match &t.class {
                TypeClassRecipe::Named { name, nu } => resolve(name, *nu as usize)
                    .ok_or_else(|| RecipeError::Element(format!("no table named {name} at nu {nu}")))?,
                TypeClassRecipe::Boolean {
                    k_log,
                    useful_bits,
                    a_0,
                    b_0,
                    c_0,
                    const_pin,
                } => TableType {
                    k_log: *k_log as usize,
                    useful_bits: usize::try_from(*useful_bits).map_err(|_| RecipeError::Range)?,
                    a_0: a_0.rebuild(),
                    b_0: b_0.rebuild(),
                    c_0: c_0.rebuild(),
                    const_pin: const_pin.map(|c| c as usize),
                    class: Default::default(),
                    io_schema: Vec::new(),
                    zeroed_word_cols: Vec::new(),
                },
                TypeClassRecipe::Element(e) => {
                    let el = ElementTableType::new(
                        e.kappa as usize,
                        e.k as usize,
                        e.a_0.rebuild(),
                        e.b_0.rebuild(),
                        e.a_const.clone(),
                        e.b_const.clone(),
                    )
                    .map_err(|err| RecipeError::Element(format!("{err:?}")))?;
                    TableType::element(Arc::new(el))
                }
            };
            ty.io_schema = t
                .io_schema
                .iter()
                .map(|&(c, out)| {
                    if out {
                        IoWord::output(c as usize)
                    } else {
                        IoWord::input(c as usize)
                    }
                })
                .collect();
            ty.zeroed_word_cols = t.zeroed_word_cols.iter().map(|&w| w as usize).collect();
            types.push(ty);
        }
        let given: Vec<TableType> = types.clone();
        let r = Registry::new(types, self.nu as usize);
        // The constructor sorts by capacity; a recipe taken from a registry
        // is already in that order, so a rebuilt one must be too: each slot
        // holds the very table given for it.
        let same = |a: &TableType, b: &TableType| {
            a.k_log == b.k_log
                && a.useful_bits == b.useful_bits
                && a.const_pin == b.const_pin
                && a.io_schema == b.io_schema
                && a.zeroed_word_cols == b.zeroed_word_cols
                && Arc::ptr_eq(&a.a_0.rows, &b.a_0.rows)
                && Arc::ptr_eq(&a.b_0.rows, &b.b_0.rows)
                && Arc::ptr_eq(&a.c_0.rows, &b.c_0.rows)
                && match (&a.class, &b.class) {
                    (TableClass::Boolean, TableClass::Boolean) => true,
                    (TableClass::LargeField(x), TableClass::LargeField(y)) => Arc::ptr_eq(x, y),
                    _ => false,
                }
        };
        if r.types().len() != given.len() || !r.types().iter().zip(&given).all(|(a, b)| same(a, b))
        {
            return Err(RecipeError::RegistryOrder);
        }
        Ok(r)
    }
}

/// A circuit: its registry, gate counts, public words, wiring classes
/// (cell indices) and the builder's optional steps.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CircuitRecipe {
    pub registry: RegistryRecipe,
    pub counts: Vec<u64>,
    pub fixed_public: Vec<Option<F128>>,
    /// The wire classes, each its cell indices `[slot | row]` (rows low,
    /// `nu` bits) ascending, classes by their least cell, as LEB128
    /// varints: per class its length, then each cell's distance from the
    /// one before it (the first cell's from the previous class's first).
    pub wires: Vec<u8>,
    pub pruned: bool,
    pub public_columns: bool,
    pub packed: bool,
    pub bus: Option<Bus>,
}

impl CircuitRecipe {
    pub fn of(c: &Circuit) -> Self {
        Self::of_with(c, &|_| None)
    }

    pub fn of_with(c: &Circuit, name: Namer<'_>) -> Self {
        Self {
            registry: RegistryRecipe::of_with(&c.registry, name),
            counts: c.counts.iter().map(|&n| n as u64).collect(),
            fixed_public: c.fixed_public.clone(),
            wires: encode_wires(&c.wires),
            pruned: !c.pruned.is_empty(),
            public_columns: !c.public_columns.is_empty(),
            packed: !c.gkr_slot.is_empty(),
            bus: c.bus.clone(),
        }
    }

    /// Run the constructors again. The caller compares the result's
    /// [`Circuit::digest`] with the one it trusts.
    pub fn rebuild(&self) -> Result<Circuit, RecipeError> {
        self.rebuild_with(&|_, _| None)
    }

    pub fn rebuild_with(&self, resolve: Resolver<'_>) -> Result<Circuit, RecipeError> {
        let registry = self.registry.rebuild_with(resolve)?;
        let nu = registry.nu();
        let counts = self
            .counts
            .iter()
            .map(|&n| usize::try_from(n).map_err(|_| RecipeError::Range))
            .collect::<Result<Vec<_>, _>>()?;
        let mask = (1usize << nu) - 1;
        let wires: Vec<Vec<Cell>> = decode_wires(&self.wires)?
            .into_iter()
            .map(|class| {
                class
                    .into_iter()
                    .map(|i| Cell {
                        slot: i >> nu,
                        row: i & mask,
                    })
                    .collect()
            })
            .collect();
        let mut c =
            Circuit::new_with_fixed_public(&registry, counts, self.fixed_public.clone(), wires)
                .map_err(RecipeError::Circuit)?;
        if self.pruned {
            c = c.with_unwired_pruned();
        }
        if self.public_columns {
            c = c.with_public_columns_opened();
        }
        if self.packed {
            c = c.with_live_slots_packed();
        }
        if let Some(bus) = &self.bus {
            let mut bus = bus.clone();
            let shared = bus.shared.take();
            c = c.with_bus(bus).map_err(RecipeError::Circuit)?;
            if let Some(s) = shared {
                if s.gamma.len() != c.bus().map_or(0, |b| b.width) {
                    return Err(RecipeError::Range);
                }
                c = c.with_shared_challenges(s);
            }
        }
        // A recipe whose optional steps did nothing (a step that finds
        // nothing to do leaves the circuit as it was) describes a different
        // circuit than it claims: refuse it.
        let fixed_ok = c.fixed_public == self.fixed_public
            && c.counts.iter().map(|&n| n as u64).eq(self.counts.iter().copied());
        let steps_ok = !c.pruned.is_empty() == self.pruned
            && !c.public_columns.is_empty() == self.public_columns
            && !c.gkr_slot.is_empty() == self.packed;
        if !fixed_ok || !steps_ok || c.bus != self.bus || encode_wires(&c.wires) != self.wires {
            return Err(RecipeError::RegistryOrder);
        }
        Ok(c)
    }
}

fn put(out: &mut Vec<u8>, mut x: u64) {
    while x >= 0x80 {
        out.push((x as u8) | 0x80);
        x >>= 7;
    }
    out.push(x as u8);
}

fn get(b: &[u8], p: &mut usize) -> Result<u64, RecipeError> {
    let mut x = 0u64;
    for shift in (0..64).step_by(7) {
        let byte = *b.get(*p).ok_or(RecipeError::Range)?;
        *p += 1;
        x |= u64::from(byte & 0x7f) << shift;
        if byte < 0x80 {
            return Ok(x);
        }
    }
    Err(RecipeError::Range)
}

fn encode_wires(wires: &[Vec<usize>]) -> Vec<u8> {
    let mut out = Vec::new();
    put(&mut out, wires.len() as u64);
    let mut first = 0usize;
    for class in wires {
        put(&mut out, class.len() as u64);
        let mut prev = first;
        for (k, &i) in class.iter().enumerate() {
            put(&mut out, (i - prev) as u64);
            if k == 0 {
                first = i;
            }
            prev = i;
        }
    }
    out
}

/// The classes back. The encoding cannot express an unsorted class; the
/// constructor checks everything else.
fn decode_wires(b: &[u8]) -> Result<Vec<Vec<usize>>, RecipeError> {
    let mut p = 0;
    let n = get(b, &mut p)? as usize;
    if n > b.len() {
        return Err(RecipeError::Range);
    }
    let mut wires = Vec::with_capacity(n);
    let mut first = 0usize;
    for _ in 0..n {
        let len = get(b, &mut p)? as usize;
        if len > b.len() {
            return Err(RecipeError::Range);
        }
        let mut class = Vec::with_capacity(len);
        let mut prev = first;
        for k in 0..len {
            let d = usize::try_from(get(b, &mut p)?).map_err(|_| RecipeError::Range)?;
            let i = prev.checked_add(d).ok_or(RecipeError::Range)?;
            if k == 0 {
                first = i;
            }
            class.push(i);
            prev = i;
        }
        wires.push(class);
    }
    if p != b.len() {
        return Err(RecipeError::Range);
    }
    Ok(wires)
}

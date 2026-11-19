//! (stub, filled by another agent)
#![allow(dead_code)]

use crate::error::Error;
use crate::tile::TileDecoder;

impl TileDecoder<'_> {
    /// Spec 5.11.34 `residual( )`: temporary stub, replaced by the coefficients agent.
    pub(crate) fn residual(&mut self) -> Result<(), Error> {
        Ok(())
    }
}

use std::{hash::Hash, marker::PhantomData};

use ahash::AHashMap;
use glam::{ivec2, ivec3, IVec2, IVec3};
use serde::{de::Visitor, ser::SerializeMap, Deserialize, Deserializer, Serialize, Serializer};

pub trait SerdeKey: Hash + Eq {
    fn to_key_string(&self) -> String;

    fn from_key_string(str: String) -> Self;
}

impl SerdeKey for IVec2 {
    fn to_key_string(&self) -> String {
        format!("{},{}", self.x, self.y)
    }

    fn from_key_string(str: String) -> Self {
        let (x, y) = str.split_once(',').unwrap();
        let (x, y) = (x.parse().unwrap(), y.parse().unwrap());
        ivec2(x, y)
    }
}

impl SerdeKey for IVec3 {
    fn to_key_string(&self) -> String {
        format!("{},{},{}", self.x, self.y, self.z)
    }

    fn from_key_string(str: String) -> Self {
        let mut parts = str.splitn(3, ',');
        let (x, y, z) = (
            parts.next().unwrap(),
            parts.next().unwrap(),
            parts.next().unwrap(),
        );
        let (x, y, z) = (x.parse().unwrap(), y.parse().unwrap(), z.parse().unwrap());
        ivec3(x, y, z)
    }
}

pub fn serialize<S: Serializer, T: Serialize, K: SerdeKey>(
    t: &AHashMap<K, T>,
    s: S,
) -> Result<S::Ok, S::Error> {
    let mut map = s.serialize_map(Some(t.len()))?;
    for (k, v) in t {
        map.serialize_entry(&k.to_key_string(), v)?;
    }
    map.end()
}

pub fn deserialize<'de, D: Deserializer<'de>, T: Deserialize<'de>, K: SerdeKey>(
    d: D,
) -> Result<AHashMap<K, T>, D::Error> {
    struct V<T, K> {
        marker: PhantomData<fn(K) -> T>,
    }
    impl<'de, T: Deserialize<'de>, K: SerdeKey> Visitor<'de> for V<T, K> {
        type Value = AHashMap<K, T>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a key-value map with stringified vectors as key")
        }

        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::MapAccess<'de>,
        {
            let mut res = AHashMap::with_capacity(map.size_hint().unwrap_or(0));
            while let Some((key, value)) = map.next_entry::<String, T>()? {
                res.insert(K::from_key_string(key), value);
            }

            Ok(res)
        }
    }
    d.deserialize_map(V {
        marker: PhantomData::default(),
    })
}
